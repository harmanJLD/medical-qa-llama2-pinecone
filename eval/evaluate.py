"""Evaluate retrieval, reranking, abstention, and end-to-end answers.

Steps:
  1. Build a question set. Either load eval/questions.jsonl, or generate questions with Llama 2
     from randomly sampled chunks (--generate N). Each question is labeled with the page it came from.
  2. Retrieval: hit@1, hit@3, and MRR for Pinecone vector search alone vs. vector search + reranking.
  3. Abstention: how often off-topic questions are correctly refused (eval/off_topic.txt).
  4. End-to-end (--answers N): answer rate, page-citation rate, citations to non-retrieved pages, latency.

Examples:
    python eval/evaluate.py --generate 40 --answers 15
    python eval/evaluate.py --questions eval/questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medqa.config import Config  # noqa: E402
from medqa.ingest import load_pdfs, split_documents  # noqa: E402

HERE = Path(__file__).parent

QUESTION_PROMPT = """[INST] <<SYS>>
You write evaluation questions. Write ONE specific question that a patient might ask and that the passage
below answers. Do not mention "the passage" or "the text". Output only the question.
<</SYS>>

Passage:
{passage} [/INST]"""


def generate_questions(llm, config: Config, n: int, seed: int = 7) -> list[dict]:
    chunks = [c for c in split_documents(load_pdfs(config.data_dir), config.chunk_size, config.chunk_overlap)
              if len(c.page_content) > 300]
    random.Random(seed).shuffle(chunks)
    questions = []
    for chunk in chunks[:n]:
        text = llm.invoke(QUESTION_PROMPT.format(passage=chunk.page_content)).strip().split("\n")[0].strip('" ')
        if text.endswith("?") and len(text) > 15:
            questions.append({"question": text, "source": chunk.metadata["source"], "page": chunk.metadata["page"]})
            print(f"  [{chunk.metadata['source']} p.{chunk.metadata['page']}] {text}")
    return questions


def rank_of(docs, q: dict) -> int | None:
    for i, d in enumerate(docs, 1):
        if d.metadata.get("page") == q["page"] and d.metadata.get("source", q["source"]) == q["source"]:
            return i
    return None


def retrieval_metrics(search, questions: list[dict]) -> dict:
    ranks = [rank_of(search(q["question"]), q) for q in questions]
    n = max(len(ranks), 1)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--generate", type=int, default=0, help="Generate N questions with Llama 2")
    parser.add_argument("--answers", type=int, default=10, help="Questions to answer end-to-end (0 to skip)")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    args = parser.parse_args()

    from medqa.chain import build_qa_chain, load_llama
    from medqa.retrieval import CrossEncoderReranker, RerankingRetriever
    from medqa.service import QAService
    from medqa.vectorstore import get_embeddings, get_vectorstore

    config = Config()
    store = get_vectorstore(config, get_embeddings(config))
    llm = load_llama(config)

    if args.generate:
        print(f"Generating {args.generate} questions with Llama 2...")
        questions = generate_questions(llm, config, args.generate)
        args.questions.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
        print(f"Saved {len(questions)} questions to {args.questions}")
    else:
        questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    off_topic = [l.strip() for l in (HERE / "off_topic.txt").read_text().splitlines() if l.strip()]

    reranker = CrossEncoderReranker(config.rerank_model)
    common = dict(vectorstore=store, fetch_k=config.fetch_k, top_k=config.top_k)
    vector_only = RerankingRetriever(reranker=None, min_vector_score=config.min_vector_score, **common)
    reranked = RerankingRetriever(reranker=reranker, min_relevance=config.min_relevance, **common)
    # Ungated copies measure ranking quality without the relevance threshold.
    vector_rank = RerankingRetriever(reranker=None, min_vector_score=-1.0, **common)
    rerank_rank = RerankingRetriever(reranker=reranker, min_relevance=-1.0, **common)

    print("Scoring retrieval...")
    results = {
        "config": {
            "questions": len(questions),
            "off_topic_questions": len(off_topic),
            "fetch_k": config.fetch_k,
            "top_k": config.top_k,
            "reranker": config.rerank_model,
            "min_relevance": config.min_relevance,
            "min_vector_score": config.min_vector_score,
        },
        "retrieval": {
            "vector_only": retrieval_metrics(vector_rank.search, questions),
            "vector_plus_rerank": retrieval_metrics(rerank_rank.search, questions),
        },
        "abstention": {
            "vector_only": {
                "off_topic_refused": round(sum(not vector_only.search(q) for q in off_topic) / len(off_topic), 3),
                "in_scope_answered": round(sum(bool(vector_only.search(q["question"])) for q in questions) / max(len(questions), 1), 3),
            },
            "vector_plus_rerank": {
                "off_topic_refused": round(sum(not reranked.search(q) for q in off_topic) / len(off_topic), 3),
                "in_scope_answered": round(sum(bool(reranked.search(q["question"])) for q in questions) / max(len(questions), 1), 3),
            },
        },
    }

    if args.answers:
        print(f"Answering {min(args.answers, len(questions))} questions end-to-end...")
        service = QAService(build_qa_chain(llm, reranked), reranked)
        answered = cited = bad_cites = 0
        latencies = []
        sample = questions[: args.answers]
        for q in sample:
            start = time.perf_counter()
            a = service.ask(q["question"])
            latencies.append(time.perf_counter() - start)
            if not a.abstained:
                answered += 1
                cited += bool(a.cited_pages)
                bad_cites += bool(a.uncited_pages)
        results["end_to_end"] = {
            "questions": len(sample),
            "answer_rate": round(answered / max(len(sample), 1), 3),
            "answers_with_page_citations": round(cited / max(answered, 1), 3),
            "answers_citing_unretrieved_pages": round(bad_cites / max(answered, 1), 3),
            "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        }

    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    r, a = results["retrieval"], results["abstention"]
    print("\n| Metric | Vector only | Vector + rerank |\n|---|---|---|")
    for key in ("hit@1", "hit@3", "mrr"):
        print(f"| Retrieval {key} | {r['vector_only'][key]} | {r['vector_plus_rerank'][key]} |")
    print(f"| Off-topic refused | {a['vector_only']['off_topic_refused']} | {a['vector_plus_rerank']['off_topic_refused']} |")
    print(f"| In-scope answered | {a['vector_only']['in_scope_answered']} | {a['vector_plus_rerank']['in_scope_answered']} |")


if __name__ == "__main__":
    main()
