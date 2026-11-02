"""Two-stage retrieval: Pinecone vector search, then cross-encoder reranking.

The retriever is a LangChain BaseRetriever, so it plugs straight into RetrievalQA.
Each returned Document carries `vector_score` and `relevance` in its metadata.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score in [0, 1] for each text."""


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly with a cross-encoder, which is more accurate than
    comparing separately computed embeddings."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self.model.predict([(query, t) for t in texts])
        return [1 / (1 + math.exp(-float(x))) for x in logits]


class RerankingRetriever(BaseRetriever):
    vectorstore: Any
    reranker: Any = None
    fetch_k: int = 10
    top_k: int = 3
    min_relevance: float = 0.05
    min_vector_score: float = 0.30

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _cache: OrderedDict = PrivateAttr(default_factory=OrderedDict)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.search(query)

    def search(self, query: str) -> list[Document]:
        """Return the top relevant chunks, or an empty list if nothing is relevant enough."""
        key = query.strip().lower()
        if key in self._cache:
            self._cache.move_to_end(key)
            return [d.model_copy(deep=True) for d in self._cache[key]]

        k = self.fetch_k if self.reranker else self.top_k
        pairs = self.vectorstore.similarity_search_with_score(query, k=k)
        docs = []
        for doc, score in pairs:
            doc.metadata["vector_score"] = round(float(score), 4)
            docs.append(doc)

        if self.reranker and docs:
            for doc, rel in zip(docs, self.reranker.score(query, [d.page_content for d in docs])):
                doc.metadata["relevance"] = round(rel, 4)
            docs.sort(key=lambda d: d.metadata["relevance"], reverse=True)
            threshold, key_name = self.min_relevance, "relevance"
        else:
            for doc in docs:
                doc.metadata["relevance"] = doc.metadata["vector_score"]
            threshold, key_name = self.min_vector_score, "vector_score"

        docs = docs[: self.top_k]
        if not docs or docs[0].metadata[key_name] < threshold:
            docs = []

        self._cache[key] = docs
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return [d.model_copy(deep=True) for d in docs]


def build_retriever(config, vectorstore, reranker: Reranker | None = None) -> RerankingRetriever:
    if reranker is None and config.use_reranker:
        reranker = CrossEncoderReranker(config.rerank_model)
    return RerankingRetriever(
        vectorstore=vectorstore,
        reranker=reranker,
        fetch_k=config.fetch_k,
        top_k=config.top_k,
        min_relevance=config.min_relevance,
        min_vector_score=config.min_vector_score,
    )
