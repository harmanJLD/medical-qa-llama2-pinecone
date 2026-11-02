"""Ask questions from the terminal.

    python scripts/ask.py "What are the symptoms of anemia?"
    python scripts/ask.py          # interactive, type q to quit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medqa.chain import ask, build_qa_chain, load_llama  # noqa: E402
from medqa.config import Config  # noqa: E402
from medqa.vectorstore import get_embeddings, get_vectorstore  # noqa: E402


def main() -> None:
    config = Config()
    store = get_vectorstore(config, get_embeddings(config))
    chain = build_qa_chain(load_llama(config), store.as_retriever(search_kwargs={"k": config.top_k}))

    def show(q: str) -> None:
        result = ask(chain, q)
        print(f"\n{result.answer}\n")
        for s in result.sources:
            print(f"  - {s['source']}, page {s['page']}")
        print()

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    while (q := input("Question: ").strip()).lower() not in {"q", "quit", "exit"}:
        if q:
            show(q)


if __name__ == "__main__":
    main()
