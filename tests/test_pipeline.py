"""Offline tests: a generated PDF, fake embeddings, an in-memory vector store, a fake reranker,
and a fake streaming LLM. No Pinecone account or model download is needed."""
import io
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import LLM
from langchain_core.vectorstores import InMemoryVectorStore

from app import Components, create_app
from medqa.chain import NO_ANSWER, PROMPT, build_qa_chain
from medqa.config import Config
from medqa.ingest import clean_page_text, load_pdfs, split_documents
from medqa.retrieval import RerankingRetriever
from medqa.service import QAService, check_citations, to_sources
from medqa.vectorstore import batch_upsert

PAGES = [
    "Iron-deficiency anemia occurs when the body lacks enough iron to make hemoglobin. "
    "Common symptoms include tiredness, pale skin, and shortness of breath. " * 6,
    "Asthma is a chronic disease in which the airways become inflamed and narrow. "
    "Triggers include pollen, dust mites, smoke, and cold air. " * 6,
]


def make_pdf(path: Path, pages: list[str]) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    for text in pages:
        y = 740
        words = text.split()
        for i in range(0, len(words), 12):
            c.drawString(50, y, " ".join(words[i : i + 12]))
            y -= 14
        c.showPage()
    c.save()
    return path


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory) -> Path:
    return make_pdf(tmp_path_factory.mktemp("data") / "reference.pdf", PAGES)


class KeywordReranker:
    """Fake cross-encoder: relevance is the share of question words found in the passage."""

    def score(self, query, texts):
        words = {w for w in query.lower().replace("?", "").split() if len(w) > 3}
        return [len(words & set(t.lower().split())) / max(len(words), 1) for t in texts]


class FakeStreamingLLM(LLM):
    """Emits a fixed answer token by token through callbacks, like LlamaCpp with streaming=True."""

    answer: str = "Symptoms include tiredness and pale skin [p. 1]."

    @property
    def _llm_type(self) -> str:
        return "fake-streaming"

    def _call(self, prompt: str, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kw: Any) -> str:
        assert "(Excerpt from reference.pdf, page" in prompt  # chunks are labeled with their page
        for token in self.answer.split(" "):
            if run_manager:
                run_manager.on_llm_new_token(token + " ")
        return self.answer


@pytest.fixture(scope="module")
def retriever(pdf_path):
    chunks = split_documents(load_pdfs(pdf_path), 300, 50)
    store = InMemoryVectorStore.from_documents(chunks, DeterministicFakeEmbedding(size=384))
    return RerankingRetriever(vectorstore=store, reranker=KeywordReranker(), fetch_k=6, top_k=2, min_relevance=0.3)


@pytest.fixture(scope="module")
def service(retriever):
    return QAService(build_qa_chain(FakeStreamingLLM(), retriever), retriever)


def test_clean_page_text():
    assert clean_page_text("hemo-\nglobin  and   iron") == "hemoglobin and iron"
    assert clean_page_text("Steps to\nmanage your\ndiabetes.") == "Steps to manage your diabetes."


def test_load_and_split(pdf_path):
    pages = load_pdfs(pdf_path)
    assert [p.metadata["page"] for p in pages] == [1, 2]
    chunks = split_documents(pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 2 and all(len(c.page_content) <= 300 for c in chunks)
    ids = [c.metadata["id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids == [c.metadata["id"] for c in split_documents(pages, 300, 50)]  # stable IDs


def test_batch_upsert(pdf_path):
    class FakeIndex:
        calls = []

        def upsert(self, vectors, namespace):
            self.calls.append((len(vectors), namespace, vectors[0]))

    chunks = split_documents(load_pdfs(pdf_path), 200, 20)
    index = FakeIndex()
    assert batch_upsert(index, chunks, DeterministicFakeEmbedding(size=384), "test", batch_size=3) == len(chunks)
    assert max(c[0] for c in index.calls) <= 3
    assert len(index.calls[0][2]["values"]) == 384
    assert {"text", "source", "page"} <= set(index.calls[0][2]["metadata"])


def test_reranker_orders_and_gates(retriever):
    docs = retriever.search("What are the symptoms of anemia like tiredness?")
    assert docs and docs[0].metadata["page"] == 1
    assert docs[0].metadata["relevance"] >= docs[-1].metadata["relevance"]
    assert retriever.search("Who won the football championship yesterday?") == []


def test_prompt_format():
    text = PROMPT.format(context="ctx", question="q")
    assert text.startswith("[INST] <<SYS>>") and text.endswith("[/INST]") and "[p. 4]" in text


def test_citation_check():
    sources = to_sources([Document(page_content="x", metadata={"source": "a", "page": 3})])
    assert check_citations("Fact [p. 3]. Other [page 9].", sources) == ([3], [9])
    assert check_citations("Fact [a.pdf, p. 3.0].", sources) == ([3], [])


def test_ask_answers_with_citations(service):
    a = service.ask("What are the symptoms of anemia like tiredness?")
    assert not a.abstained and a.cited_pages == [1] and a.uncited_pages == []
    assert a.sources[0].page == 1 and a.timings["generation_s"] >= 0


def test_ask_abstains_without_calling_model(service):
    a = service.ask("Who won the football championship yesterday?")
    assert a.abstained and a.answer == NO_ANSWER and "generation_s" not in a.timings


def test_stream_events(service):
    events = list(service.stream("What are the symptoms of anemia like tiredness?"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and kinds.count("token") > 3
    assert events[-1]["cited_pages"] == [1]


def test_flask_routes(service, tmp_path):
    config = Config()
    config.data_dir = str(tmp_path)
    client = create_app(Components(service, config=config)).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["status"] == "ok"

    ok = client.post("/api/ask", json={"question": "What are the symptoms of anemia like tiredness?"})
    assert ok.status_code == 200 and ok.get_json()["cited_pages"] == [1]
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400

    stream = client.post("/api/stream", json={"question": "What are the symptoms of anemia like tiredness?"})
    events = [json.loads(l[6:]) for l in stream.get_data(as_text=True).split("\n\n") if l.startswith("data: ")]
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"

    assert client.get("/api/documents").get_json() == {"documents": []}
    bad = client.post("/api/upload", data={"file": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
    assert bad.status_code == 400


def test_upload_indexes_pdf(service, tmp_path):
    class FakeIndex:
        count = 0

        def upsert(self, vectors, namespace):
            FakeIndex.count += len(vectors)

    config = Config()
    config.data_dir = str(tmp_path)
    comps = Components(service, index=FakeIndex(), embeddings=DeterministicFakeEmbedding(size=384), config=config)
    client = create_app(comps).test_client()
    pdf = make_pdf(tmp_path / "src.pdf", PAGES).read_bytes()
    (tmp_path / "src.pdf").unlink()
    res = client.post("/api/upload", data={"file": (io.BytesIO(pdf), "new guide.pdf")}, content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 200 and body["pages"] == 2 and body["vectors"] == FakeIndex.count > 0
    assert client.get("/api/documents").get_json()["documents"][0]["name"] == "new_guide.pdf"


def test_json_and_scores_handle_nan():
    import math

    from app import to_json
    from medqa.retrieval import sigmoid

    assert json.loads(to_json({"relevance": float("nan"), "x": [float("inf"), 1.5]})) == {"relevance": None, "x": [None, 1.5]}
    assert math.isnan(sigmoid(float("nan"))) and sigmoid(-1000) < 1e-9 and math.isclose(sigmoid(0), 0.5)


def test_broken_reranker_falls_back_to_vector_search(retriever):
    class NaNReranker:
        def score(self, query, texts):
            return [float("nan")] * len(texts)

    broken = RerankingRetriever(vectorstore=retriever.vectorstore, reranker=NaNReranker(), fetch_k=6, top_k=2,
                                min_relevance=0.3, min_vector_score=-1.0)
    docs = broken.search("What are the symptoms of anemia like tiredness?")
    assert docs and all(d.metadata["relevance"] == d.metadata["vector_score"] for d in docs)
