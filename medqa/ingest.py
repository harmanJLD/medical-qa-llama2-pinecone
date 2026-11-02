"""Load medical reference PDFs, clean the text, and split it into chunks with LangChain."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SPACES = re.compile(r"[ \t]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![.\n:])\n(?!\n|[•\-\d])")


def clean_page_text(text: str) -> str:
    """Repair PDF extraction artifacts: hyphenated line breaks, mid-sentence line breaks, extra spaces."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_BREAK.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def load_pdfs(path: str | Path) -> list[Document]:
    """Load one PDF or every PDF in a folder. Each page becomes a Document with source and page metadata."""
    path = Path(path)
    files = [path] if path.suffix.lower() == ".pdf" else sorted(path.glob("**/*.pdf"))
    if not files:
        raise FileNotFoundError(f"No PDF files found in {path}")

    pages: list[Document] = []
    for pdf in files:
        for page in PyPDFLoader(str(pdf)).load():
            text = clean_page_text(page.page_content)
            if len(text) < 50:  # skip blank, cover, and image-only pages
                continue
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page": int(page.metadata.get("page", 0)) + 1},
                )
            )
    if not pages:
        raise ValueError(f"No extractable text in {path}. Scanned PDFs need OCR first.")
    return pages


def split_documents(pages: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
        chunk.metadata["id"] = chunk_id(chunk)
    return chunks


def chunk_id(chunk: Document) -> str:
    """Stable ID from source, page, and text, so re-running ingestion overwrites instead of duplicating."""
    key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{chunk.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def ingest(path: str | Path, index, embeddings, config) -> dict:
    """Full pipeline for one PDF or folder: load, chunk, embed, batch-upsert. Returns a summary."""
    from .vectorstore import batch_upsert

    pages = load_pdfs(path)
    chunks = split_documents(pages, config.chunk_size, config.chunk_overlap)
    n = batch_upsert(index, chunks, embeddings, config.namespace, config.batch_size)
    return {"pages": len(pages), "chunks": len(chunks), "vectors": n}
