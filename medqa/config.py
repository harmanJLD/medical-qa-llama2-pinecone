"""Settings loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


@dataclass
class Config:
    pinecone_api_key: str = field(default_factory=lambda: _get("PINECONE_API_KEY", ""))
    index_name: str = field(default_factory=lambda: _get("PINECONE_INDEX", "medical-qa"))
    cloud: str = field(default_factory=lambda: _get("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: _get("PINECONE_REGION", "us-east-1"))
    namespace: str = field(default_factory=lambda: _get("PINECONE_NAMESPACE", "default"))

    embed_model: str = field(
        default_factory=lambda: _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embed_dim: int = 384  # all-MiniLM-L6-v2 output size

    llama_model_path: str = field(
        default_factory=lambda: _get("LLAMA_MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
    )
    llama_n_ctx: int = field(default_factory=lambda: int(_get("LLAMA_N_CTX", "4096")))
    llama_max_tokens: int = field(default_factory=lambda: int(_get("LLAMA_MAX_TOKENS", "512")))
    llama_temperature: float = field(default_factory=lambda: float(_get("LLAMA_TEMPERATURE", "0.2")))
    llama_gpu_layers: int = field(default_factory=lambda: int(_get("LLAMA_GPU_LAYERS", "0")))

    data_dir: str = field(default_factory=lambda: _get("DATA_DIR", "data"))
    chunk_size: int = field(default_factory=lambda: int(_get("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_get("CHUNK_OVERLAP", "100")))
    top_k: int = field(default_factory=lambda: int(_get("TOP_K", "3")))
    fetch_k: int = field(default_factory=lambda: int(_get("FETCH_K", "10")))
    rerank_model: str = field(
        default_factory=lambda: _get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    use_reranker: bool = field(default_factory=lambda: _get("USE_RERANKER", "true").lower() == "true")
    # Minimum relevance (0-1) of the best chunk before the model is asked to answer.
    min_relevance: float = field(default_factory=lambda: float(_get("MIN_RELEVANCE", "0.05")))
    # Used when the reranker is off: minimum Pinecone cosine similarity.
    min_vector_score: float = field(default_factory=lambda: float(_get("MIN_VECTOR_SCORE", "0.30")))
    batch_size: int = 100

    def require_pinecone(self) -> None:
        if not self.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set. Copy .env.example to .env and add your key.")
