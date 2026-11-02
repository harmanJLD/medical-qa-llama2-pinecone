"""Flask web app for the medical Q&A chain.

    python app.py      # http://127.0.0.1:5000
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from medqa.chain import ask, build_qa_chain, load_llama
from medqa.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("medqa")

MAX_QUESTION_CHARS = 1000


def create_app(chain=None) -> Flask:
    """Build the app. Pass a chain for testing; otherwise it is loaded on the first question."""
    app = Flask(__name__)
    state = {"chain": chain}

    def get_chain():
        if state["chain"] is None:
            from medqa.vectorstore import get_embeddings, get_vectorstore

            config = Config()
            log.info("Loading embeddings, Pinecone index, and Llama 2 model...")
            store = get_vectorstore(config, get_embeddings(config))
            retriever = store.as_retriever(search_kwargs={"k": config.top_k})
            state["chain"] = build_qa_chain(load_llama(config), retriever)
        return state["chain"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/ask")
    def api_ask():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify(error="Please enter a question."), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify(error=f"Questions are limited to {MAX_QUESTION_CHARS} characters."), 400
        try:
            result = ask(get_chain(), question)
        except Exception as exc:
            log.exception("Failed to answer question")
            return jsonify(error=f"Could not answer right now: {exc}"), 500
        log.info("Answered question with %d sources", len(result.sources))
        return jsonify(result.to_dict())

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
