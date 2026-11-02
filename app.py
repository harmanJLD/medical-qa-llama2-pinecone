"""Flask web app: streaming medical Q&A with page citations and PDF upload.

    python app.py      # http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

from medqa.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("medqa")

MAX_QUESTION_CHARS = 1000
MAX_UPLOAD_MB = 50


class Components:
    """Everything the app needs at runtime. Built once at startup (or injected in tests)."""

    def __init__(self, service, index=None, embeddings=None, config: Config | None = None):
        self.service = service
        self.index = index
        self.embeddings = embeddings
        self.config = config or Config()

    @classmethod
    def load(cls, config: Config) -> "Components":
        from medqa.chain import build_qa_chain, load_llama
        from medqa.retrieval import build_retriever
        from medqa.service import QAService
        from medqa.vectorstore import get_embeddings, get_pinecone_index, get_vectorstore

        t = time.perf_counter()
        log.info("Loading embedding model %s", config.embed_model)
        embeddings = get_embeddings(config)
        log.info("Connecting to Pinecone index '%s'", config.index_name)
        index = get_pinecone_index(config, create=False)
        store = get_vectorstore(config, embeddings)
        log.info("Loading reranker %s", config.rerank_model if config.use_reranker else "(disabled)")
        retriever = build_retriever(config, store)
        log.info("Loading Llama 2 from %s (gpu layers: %s)", config.llama_model_path, config.llama_gpu_layers)
        chain = build_qa_chain(load_llama(config), retriever)
        log.info("Ready in %.1fs. Open http://127.0.0.1:5000", time.perf_counter() - t)
        return cls(QAService(chain, retriever), index, embeddings, config)


def create_app(components: Components) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    data_dir = Path(components.config.data_dir)

    def read_question() -> tuple[str | None, tuple | None]:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return None, (jsonify(error="Please enter a question."), 400)
        if len(question) > MAX_QUESTION_CHARS:
            return None, (jsonify(error=f"Questions are limited to {MAX_QUESTION_CHARS} characters."), 400)
        return question, None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        c = components.config
        return jsonify(
            status="ok",
            index=c.index_name,
            reranker=c.rerank_model if c.use_reranker else None,
            top_k=c.top_k,
            fetch_k=c.fetch_k,
        )

    @app.post("/api/ask")
    def ask():
        question, error = read_question()
        if error:
            return error
        try:
            result = components.service.ask(question)
        except Exception as exc:
            log.exception("Failed to answer")
            return jsonify(error=f"Could not answer: {exc}"), 500
        return jsonify(result.to_dict())

    @app.post("/api/stream")
    def stream():
        question, error = read_question()
        if error:
            return error

        def events():
            try:
                for event in components.service.stream(question):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                log.exception("Streaming failed")
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/documents")
    def documents():
        files = sorted(data_dir.glob("*.pdf")) if data_dir.exists() else []
        return jsonify(documents=[{"name": f.name, "size_kb": round(f.stat().st_size / 1024)} for f in files])

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify(error="Choose a PDF to upload."), 400
        name = secure_filename(file.filename)
        if not name.lower().endswith(".pdf"):
            return jsonify(error="Only PDF files are supported."), 400
        if components.index is None:
            return jsonify(error="Uploads are not available in this mode."), 503
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / name
        file.save(path)
        try:
            from medqa.ingest import ingest

            summary = ingest(path, components.index, components.embeddings, components.config)
        except Exception as exc:
            path.unlink(missing_ok=True)
            log.exception("Upload failed")
            return jsonify(error=f"Could not index {name}: {exc}"), 400
        components.service.retriever._cache.clear()
        log.info("Indexed %s: %s", name, summary)
        return jsonify(name=name, **summary)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"File is larger than {MAX_UPLOAD_MB} MB."), 413

    return app


if __name__ == "__main__":
    create_app(Components.load(Config())).run(host="127.0.0.1", port=5000, threaded=True)
