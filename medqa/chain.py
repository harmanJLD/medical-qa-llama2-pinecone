"""Quantized Llama 2 model and the RetrievalQA chain with page-cited answers."""
from __future__ import annotations

from pathlib import Path

from langchain_classic.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from .config import Config

# Llama 2 chat format: [INST] <<SYS>> system <</SYS>> user [/INST]
PROMPT_TEMPLATE = """[INST] <<SYS>>
You are a medical information assistant. Answer using only the excerpts below.
After each fact, cite the page it came from in square brackets, for example [p. 4].
If the excerpts do not answer the question, reply only: "I don't know based on the available reference."
Do not make up facts. Keep the answer under 150 words. This is general information, not medical advice.
<</SYS>>

Excerpts:
{context}

Question: {question} [/INST]"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# How each retrieved chunk is shown to the model, so it can cite pages.
DOCUMENT_PROMPT = PromptTemplate(
    template="[{source}, p. {page}]\n{page_content}",
    input_variables=["source", "page", "page_content"],
)

NO_ANSWER = "I don't know based on the available reference."


def load_llama(config: Config) -> BaseLLM:
    """Load the quantized Llama 2 7B chat model (GGUF) with llama.cpp, with token streaming enabled."""
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
        n_batch=512,
        streaming=True,
        stop=["[INST]", "</s>"],
        verbose=False,
    )


def build_qa_chain(llm: BaseLLM, retriever: BaseRetriever) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT, "document_prompt": DOCUMENT_PROMPT},
    )
