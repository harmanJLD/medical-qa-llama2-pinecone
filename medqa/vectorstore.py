"""Embeddings and the Pinecone serverless index."""
from __future__ import annotations

import time
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from .config import Config


def get_embeddings(config: Config) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_pinecone_index(config: Config, create: bool = False):
    """Connect to the Pinecone index, creating a serverless index on AWS if requested."""
    from pinecone import Pinecone, ServerlessSpec

    config.require_pinecone()
    pc = Pinecone(api_key=config.pinecone_api_key)
    if not pc.has_index(config.index_name):
        if not create:
            raise RuntimeError(f"Pinecone index '{config.index_name}' not found. Run: python scripts/ingest.py")
        pc.create_index(
            name=config.index_name,
            dimension=config.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.cloud, region=config.region),
        )
        while not pc.describe_index(config.index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(config.index_name)


def _batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def batch_upsert(index, chunks: list[Document], embeddings: Embeddings, namespace: str, batch_size: int = 100) -> int:
    """Embed chunks and upsert them to Pinecone in batches. Returns the number of vectors written."""
    written = 0
    for batch in tqdm(list(_batches(chunks, batch_size)), desc="Upserting", unit="batch"):
        vectors = embeddings.embed_documents([c.page_content for c in batch])
        records = [
            {
                "id": c.metadata["id"],
                "values": vec,
                "metadata": {
                    "text": c.page_content,
                    "source": c.metadata.get("source", ""),
                    "page": c.metadata.get("page", 0),
                },
            }
            for c, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=records, namespace=namespace)
        written += len(records)
    return written


def get_vectorstore(config: Config, embeddings: Embeddings):
    from langchain_pinecone import PineconeVectorStore

    index = get_pinecone_index(config, create=False)
    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text", namespace=config.namespace)
