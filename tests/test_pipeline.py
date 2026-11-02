"""Offline tests: a generated PDF, fake embeddings, an in-memory vector store, and a fake LLM.
No Pinecone account or model download is needed."""
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListLLM
from langchain_core.vectorstores import InMemoryVectorStore

from app import create_app
from medqa.chain import PROMPT, ask, build_qa_chain
from medqa.ingest import clean_page_text, load_pdfs, split_documents
from medqa.vectorstore import batch_upsert

PAGES = [
    "Iron-deficiency anemia occurs when the body lacks enough iron to make hemoglobin. "
    "Common symptoms include tiredness, pale skin, and shortness of breath. " * 6,
    "Asthma is a chronic disease in which the airways become inflamed and narrow. "
    "Triggers include pollen, dust mites, smoke, and cold air. " * 6,
]


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path_factory.mktemp("data") / "reference.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    for text in PAGES:
        y = 740
        words = text.split()
        for i in range(0, len(words), 12):
            c.drawString(50, y, " ".join(words[i : i + 12]))
            y -= 14
        c.showPage()
    c.save()
    return path


def test_clean_page_text():
    assert clean_page_text("hemo-\nglobin  and   iron") == "hemoglobin and iron"


def test_load_and_split(pdf_path):
    pages = load_pdfs(pdf_path)
    assert [p.metadata["page"] for p in pages] == [1, 2]
    chunks = split_documents(pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 2
    assert all(len(c.page_content) <= 300 for c in chunks)
    ids = [c.metadata["id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    # Same input gives the same IDs, so re-ingesting overwrites instead of duplicating.
    assert ids == [c.metadata["id"] for c in split_documents(pages, 300, 50)]


class FakeIndex:
    def __init__(self):
        self.calls = []

    def upsert(self, vectors, namespace):
        self.calls.append((len(vectors), namespace, vectors[0]))


def test_batch_upsert(pdf_path):
    chunks = split_documents(load_pdfs(pdf_path), 200, 20)
    index = FakeIndex()
    n = batch_upsert(index, chunks, DeterministicFakeEmbedding(size=384), "test", batch_size=3)
    assert n == len(chunks)
    assert sum(c[0] for c in index.calls) == len(chunks)
    assert max(c[0] for c in index.calls) <= 3
    first = index.calls[0][2]
    assert len(first["values"]) == 384
    assert {"text", "source", "page"} <= set(first["metadata"])


@pytest.fixture(scope="module")
def chain(pdf_path):
    chunks = split_documents(load_pdfs(pdf_path), 300, 50)
    store = InMemoryVectorStore.from_documents(chunks, DeterministicFakeEmbedding(size=384))
    llm = FakeListLLM(responses=["Symptoms include tiredness and pale skin."] * 10)
    return build_qa_chain(llm, store.as_retriever(search_kwargs={"k": 2}))


def test_prompt_uses_llama2_format():
    text = PROMPT.format(context="ctx", question="q")
    assert text.startswith("[INST] <<SYS>>") and text.endswith("[/INST]")


def test_ask_returns_answer_and_sources(chain):
    result = ask(chain, "What are the symptoms of anemia?")
    assert result.answer == "Symptoms include tiredness and pale skin."
    assert result.sources and result.sources[0]["source"] == "reference.pdf"
    assert result.sources[0]["page"] in (1, 2)


def test_flask_endpoint(chain):
    client = create_app(chain=chain).test_client()
    assert client.get("/").status_code == 200
    ok = client.post("/api/ask", json={"question": "What triggers asthma?"})
    assert ok.status_code == 200 and ok.get_json()["answer"]
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 400
