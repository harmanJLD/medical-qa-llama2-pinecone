"""Ingest PDFs into Pinecone: load, chunk, embed, and batch-upsert.

    python scripts/ingest.py                    # every PDF in data/
    python scripts/ingest.py --pdf data/ref.pdf # a single PDF
    python scripts/ingest.py --dry-run          # chunk only, no Pinecone calls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medqa.config import Config  # noqa: E402
from medqa.ingest import load_pdfs, split_documents  # noqa: E402


def main() -> None:
    config = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default=config.data_dir, help="PDF file or folder")
    parser.add_argument("--dry-run", action="store_true", help="Only load and chunk")
    args = parser.parse_args()

    pages = load_pdfs(args.pdf)
    chunks = split_documents(pages, config.chunk_size, config.chunk_overlap)
    print(f"Loaded {len(pages)} pages and created {len(chunks)} chunks.")
    if args.dry_run:
        for c in chunks[:3]:
            print(f"\n[{c.metadata['source']} p.{c.metadata['page']}] {c.page_content[:200]}...")
        return

    from medqa.vectorstore import batch_upsert, get_embeddings, get_pinecone_index

    index = get_pinecone_index(config, create=True)
    n = batch_upsert(index, chunks, get_embeddings(config), config.namespace, config.batch_size)
    print(f"Upserted {n} vectors to Pinecone index '{config.index_name}' (namespace '{config.namespace}').")


if __name__ == "__main__":
    main()
