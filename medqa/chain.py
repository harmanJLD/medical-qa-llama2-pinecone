"""Quantized Llama 2 model and the RetrievalQA chain."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_classic.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from .config import Config

# Llama 2 chat format: [INST] <<SYS>> system <</SYS>> user [/INST]
PROMPT_TEMPLATE = """[INST] <<SYS>>
You are a medical information assistant. Answer the question using only the context below.
If the context does not contain the answer, say "I don't know based on the available reference."
Do not make up facts. Keep the answer clear and under 150 words. This is general information, not medical advice.
<</SYS>>

Context:
{context}

Question: {question} [/INST]"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def load_llama(config: Config) -> BaseLLM:
    """Load the quantized Llama 2 7B chat model (GGUF) with llama.cpp."""
    from langchain_community.llms import LlamaCpp

    model_path = Path(config.llama_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. See models/README.md.")
    return LlamaCpp(
        model_path=str(model_path),
        n_ctx=config.llama_n_ctx,
        max_tokens=config.llama_max_tokens,
        temperature=config.llama_temperature,
        n_gpu_layers=config.llama_gpu_layers,
        stop=["[INST]"],
        verbose=False,
    )


def build_qa_chain(llm: BaseLLM, retriever: BaseRetriever) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"answer": self.answer, "sources": self.sources}


def ask(chain: RetrievalQA, question: str) -> Answer:
    result = chain.invoke({"query": question})
    seen, sources = set(), []
    for doc in result.get("source_documents", []):
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page"),
                "excerpt": doc.page_content[:300],
            }
        )
    return Answer(answer=result["result"].strip(), sources=sources)
