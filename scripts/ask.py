"""Ask questions from the terminal, with streamed answers.

    python scripts/ask.py "What is a good A1C goal?"
    python scripts/ask.py          # interactive, type q to quit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medqa.chain import build_qa_chain, load_llama  # noqa: E402
from medqa.config import Config  # noqa: E402
from medqa.retrieval import build_retriever  # noqa: E402
from medqa.service import QAService  # noqa: E402
from medqa.vectorstore import get_embeddings, get_vectorstore  # noqa: E402


def main() -> None:
    config = Config()
    retriever = build_retriever(config, get_vectorstore(config, get_embeddings(config)))
    service = QAService(build_qa_chain(load_llama(config), retriever), retriever)

    def show(question: str) -> None:
        print()
        for event in service.stream(question):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "error":
                print(f"Error: {event['message']}")
            elif event["type"] == "done":
                if event["abstained"]:
                    print(event["answer"])
                print("\n")
                for s in event["sources"]:
                    print(f"  - {s['source']}, p. {s['page']} (relevance {s['relevance']})")
                print(f"  timings: {event['timings']}\n")

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    while (q := input("Question: ").strip()).lower() not in {"q", "quit", "exit"}:
        if q:
            show(q)


if __name__ == "__main__":
    main()
