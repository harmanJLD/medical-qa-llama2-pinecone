"""Two-stage retrieval: Pinecone vector search, then cross-encoder reranking.

The retriever is a LangChain BaseRetriever, so it plugs straight into RetrievalQA.
Each returned Document carries `vector_score` and `relevance` in its metadata.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Any, Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

log = logging.getLogger("medqa")


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score in [0, 1] for each text."""


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly with a cross-encoder, which is more accurate than
    comparing separately computed embeddings."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        import inspect

        import torch
        from sentence_transformers import CrossEncoder

        # Try the default setup first. If it produces NaN scores (seen with some torch/transformers
        # versions on macOS), retry with eager attention in float32, which avoids the fused kernels.
        attempts = [
            ("default", {}),
            ("eager attention, float32", {"model_kwargs": {"attn_implementation": "eager", "torch_dtype": torch.float32}}),
        ]
        self.healthy = False
        for label, extra in attempts:
            try:
                # CPU is fast enough for 10 short passages and avoids NaN outputs seen on Apple's MPS backend.
                self.model = CrossEncoder(model_name, device="cpu", **extra)
            except TypeError:  # older sentence-transformers without model_kwargs
                continue
            # Newer sentence-transformers versions apply a sigmoid inside predict(). Ask for raw logits
            # so the sigmoid is applied exactly once, below.
            params = inspect.signature(self.model.predict).parameters
            self._predict_kwargs = {}
            for name in ("activation_fn", "activation_fct"):
                if name in params:
                    self._predict_kwargs[name] = torch.nn.Identity()
                    break
            log.info("Testing reranker (%s)", label)
            if self._self_test():
                self.healthy = True
                break

    def _self_test(self) -> bool:
        """Check the model gives finite scores that rank an obvious match above a non-match."""
        good, bad = self.score(
            "What is a normal body temperature?",
            ["Normal body temperature is about 98.6 degrees Fahrenheit (37 degrees Celsius).",
             "The Eiffel Tower is a wrought-iron tower in Paris, France."],
        )
        ok = math.isfinite(good) and math.isfinite(bad) and good > bad
        if ok:
            log.info("Reranker self-test passed (match %.3f, non-match %.3f)", good, bad)
        else:
            log.warning("Reranker self-test FAILED (match %s, non-match %s); using vector search only", good, bad)
        return ok

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self.model.predict([(query, t) for t in texts], **self._predict_kwargs)
        return [sigmoid(x) for x in logits]


def sigmoid(x) -> float:
    """Logit to probability. NaN stays NaN so callers can detect a broken model."""
    x = float(x)
    if math.isnan(x):
        return float("nan")
    x = max(-30.0, min(30.0, x))
    return 1 / (1 + math.exp(-x))


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
            doc.metadata["vector_score"] = _finite(score)
            if "page" in doc.metadata:  # Pinecone returns numbers as floats (7.0)
                doc.metadata["page"] = int(float(doc.metadata["page"]))
            docs.append(doc)

        scores = self.reranker.score(query, [d.page_content for d in docs]) if self.reranker and docs else []
        if scores and not all(math.isfinite(x) for x in scores):
            log.warning("Reranker returned invalid scores; falling back to vector search for this query")
            scores = []

        if scores:
            for doc, rel in zip(docs, scores):
                doc.metadata["relevance"] = round(rel, 4)
            docs.sort(key=lambda d: d.metadata["relevance"], reverse=True)
            threshold, key_name = self.min_relevance, "relevance"
        else:
            for doc in docs:
                doc.metadata["relevance"] = doc.metadata["vector_score"]
            threshold, key_name = self.min_vector_score, "vector_score"

        docs = docs[: self.top_k]
        if docs:
            log.info("Query %r: best %s %.3f (threshold %.2f)", query[:60], key_name, docs[0].metadata[key_name], threshold)
        else:
            log.info("Query %r: Pinecone returned no matches", query[:60])
        if not docs or docs[0].metadata[key_name] < threshold:
            docs = []

        self._cache[key] = docs
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return [d.model_copy(deep=True) for d in docs]


def _finite(x) -> float:
    x = float(x)
    return round(x, 4) if math.isfinite(x) else 0.0


def build_retriever(config, vectorstore, reranker: Reranker | None = None) -> RerankingRetriever:
    if reranker is None and config.use_reranker:
        candidate = CrossEncoderReranker(config.rerank_model)
        reranker = candidate if candidate.healthy else None
    return RerankingRetriever(
        vectorstore=vectorstore,
        reranker=reranker,
        fetch_k=config.fetch_k,
        top_k=config.top_k,
        min_relevance=config.min_relevance,
        min_vector_score=config.min_vector_score,
    )
