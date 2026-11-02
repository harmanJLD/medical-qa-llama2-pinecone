"""Question-answering service: relevance gate, RetrievalQA, citations, streaming, and timing."""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator

from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

from .chain import NO_ANSWER
from .retrieval import RerankingRetriever

_PAGE_CITE = re.compile(r"\[p(?:age|\.)?\s*(\d+)\]", re.IGNORECASE)


@dataclass
class Source:
    source: str
    page: int
    excerpt: str
    relevance: float | None = None
    vector_score: float | None = None


@dataclass
class Answer:
    answer: str
    sources: list[Source] = field(default_factory=list)
    abstained: bool = False
    cited_pages: list[int] = field(default_factory=list)
    uncited_pages: list[int] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def to_sources(docs: list[Document]) -> list[Source]:
    sources, seen = [], set()
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                source=d.metadata.get("source", ""),
                page=int(d.metadata.get("page", 0)),
                excerpt=d.page_content.lstrip(" .,;:")[:400],
                relevance=d.metadata.get("relevance"),
                vector_score=d.metadata.get("vector_score"),
            )
        )
    return sources


def check_citations(answer: str, sources: list[Source]) -> tuple[list[int], list[int]]:
    """Split cited pages into ones that were retrieved and ones that were not (possible hallucination)."""
    cited = sorted({int(p) for p in _PAGE_CITE.findall(answer)})
    retrieved = {s.page for s in sources}
    return [p for p in cited if p in retrieved], [p for p in cited if p not in retrieved]


class _TokenQueue(BaseCallbackHandler):
    def __init__(self, q: queue.Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(("token", token))


class QAService:
    def __init__(self, chain: RetrievalQA, retriever: RerankingRetriever):
        self.chain = chain
        self.retriever = retriever
        self._lock = threading.Lock()  # llama.cpp models are not safe for concurrent calls

    def _retrieve(self, question: str) -> tuple[list[Document], float]:
        start = time.perf_counter()
        docs = self.retriever.search(question)
        return docs, round(time.perf_counter() - start, 3)

    def _finish(self, text: str, sources: list[Source], timings: dict) -> Answer:
        text = text.strip()
        if NO_ANSWER.lower().rstrip(".") in text.lower():
            return Answer(answer=NO_ANSWER, sources=[], abstained=True, timings=timings)
        cited, uncited = check_citations(text, sources)
        return Answer(text, sources, False, cited, uncited, timings)

    def ask(self, question: str) -> Answer:
        docs, t_retrieve = self._retrieve(question)
        if not docs:
            return Answer(answer=NO_ANSWER, abstained=True, timings={"retrieval_s": t_retrieve})
        start = time.perf_counter()
        with self._lock:
            result = self.chain.invoke({"query": question})
        timings = {"retrieval_s": t_retrieve, "generation_s": round(time.perf_counter() - start, 3)}
        return self._finish(result["result"], to_sources(result["source_documents"]), timings)

    def stream(self, question: str) -> Iterator[dict]:
        """Yield events: sources, then tokens as they are generated, then the final answer."""
        docs, t_retrieve = self._retrieve(question)
        sources = to_sources(docs)
        if not docs:
            yield {"type": "done", **Answer(NO_ANSWER, abstained=True, timings={"retrieval_s": t_retrieve}).to_dict()}
            return
        yield {"type": "sources", "sources": [asdict(s) for s in sources]}

        q: queue.Queue = queue.Queue()
        start = time.perf_counter()
        first_token: list[float] = []

        def run() -> None:
            try:
                with self._lock:
                    result = self.chain.invoke({"query": question}, config={"callbacks": [_TokenQueue(q)]})
                q.put(("result", result["result"]))
            except Exception as exc:  # report errors to the client instead of hanging
                q.put(("error", str(exc)))

        threading.Thread(target=run, daemon=True).start()
        streamed = []
        while True:
            kind, value = q.get()
            if kind == "token":
                if not first_token:
                    first_token.append(time.perf_counter() - start)
                streamed.append(value)
                yield {"type": "token", "text": value}
            elif kind == "error":
                yield {"type": "error", "message": value}
                return
            else:
                text = value or "".join(streamed)
                timings = {
                    "retrieval_s": t_retrieve,
                    "first_token_s": round(first_token[0], 3) if first_token else None,
                    "generation_s": round(time.perf_counter() - start, 3),
                }
                yield {"type": "done", **self._finish(text, sources, timings).to_dict()}
                return
