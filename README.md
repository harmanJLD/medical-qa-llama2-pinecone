# Medical Q&A with Llama 2 and Pinecone

A retrieval-augmented medical question-answering app. It ingests a medical reference PDF, chunks it with LangChain, embeds the chunks with Hugging Face `all-MiniLM-L6-v2`, batch-upserts the vectors to a Pinecone serverless index on AWS, and answers questions with a quantized Llama 2 7B chat model through a LangChain RetrievalQA chain served by a Flask web app. Every answer lists the PDF pages it was based on.

Author: Harmandeep Kaur

## Architecture

```
Ingestion
  PDF ──► PyPDFLoader ──► clean text ──► RecursiveCharacterTextSplitter (800 chars, 100 overlap)
      ──► all-MiniLM-L6-v2 embeddings (384-d) ──► batch upsert (100/batch) ──► Pinecone serverless (AWS)

Question answering
  question ──► Pinecone similarity search (top 3) ──► Llama 2 chat prompt ──► Llama 2 7B (Q4_K_M, llama.cpp)
           ──► answer + source pages ──► Flask JSON API ──► chat UI
```

## Design choices

- **Page-level citations.** Each chunk keeps its source file and page number in Pinecone metadata, and the app returns them with every answer.
- **Idempotent ingestion.** Vector IDs are a SHA-1 hash of source, page, and text, so re-running ingestion updates vectors instead of duplicating them.
- **Text cleanup.** Hyphenated line breaks and extra spaces from PDF extraction are repaired, and near-empty pages are skipped.
- **Llama 2 chat prompt format.** The prompt uses `[INST] <<SYS>>` tags, tells the model to answer only from the context, and to say "I don't know" when the context lacks the answer.
- **Local quantized model.** Llama 2 7B chat in 4-bit GGUF (about 4 GB) runs on a laptop CPU with llama.cpp. GPU offloading is optional.
- **Configuration.** All settings (keys, model path, chunk size, top-k, region) come from `.env`, with no hardcoded paths.
- **Offline tests.** The test suite uses a generated PDF, fake embeddings, an in-memory vector store, and a fake LLM, so it runs without a Pinecone account or model download.

## Project structure

| Path | Purpose |
|---|---|
| `medqa/config.py` | Settings from environment variables |
| `medqa/ingest.py` | PDF loading, cleanup, chunking, stable chunk IDs |
| `medqa/vectorstore.py` | Embeddings, Pinecone serverless index creation, batch upsert |
| `medqa/chain.py` | Llama 2 loader, prompt, RetrievalQA chain, answer formatting |
| `scripts/ingest.py` | Command-line ingestion |
| `scripts/ask.py` | Command-line question answering |
| `app.py` | Flask app and JSON API |
| `templates/`, `static/` | Chat interface |
| `tests/` | Offline tests |

## Setup

**1. Install**

```bash
git clone https://github.com/harmanJLD/medical-qa-llama2-pinecone.git
cd medical-qa-llama2-pinecone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Apple Silicon, install llama.cpp with Metal support for faster answers:
`CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python`

**2. Add your keys and files**

- Create a free Pinecone account and put the API key in `.env` as `PINECONE_API_KEY`.
- Download the model into `models/` (see `models/README.md`):
  `hf download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir models`
- Put a medical reference PDF in `data/`. Public-domain publications from NIH, CDC, or NCI work well.

**3. Ingest**

```bash
python scripts/ingest.py --dry-run   # check chunking first
python scripts/ingest.py             # creates the serverless index and upserts vectors
```

**4. Run**

```bash
python app.py                        # http://127.0.0.1:5000
python scripts/ask.py "What are the symptoms of iron-deficiency anemia?"
```

## API

`POST /api/ask` with `{"question": "..."}` returns:

```json
{
  "answer": "Iron-deficiency anemia can cause tiredness, pale skin, and shortness of breath...",
  "sources": [{"source": "reference.pdf", "page": 117, "excerpt": "..."}]
}
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PINECONE_INDEX` | `medical-qa` | Index name |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless location |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLAMA_MODEL_PATH` | `models/llama-2-7b-chat.Q4_K_M.gguf` | Quantized model file |
| `LLAMA_GPU_LAYERS` | `0` | Layers to offload to GPU (`-1` for all) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Chunking, in characters |
| `TOP_K` | `3` | Chunks retrieved per question |

## Tests

```bash
pytest
```

## Limitations

- Answers are only as good as the indexed PDF, and a 7B model can still misread the context.
- Scanned PDFs without a text layer need OCR before ingestion.
- CPU generation takes several seconds per answer.

## Disclaimer

For research and education only. Not a medical device, and not a substitute for professional medical advice.

## License

MIT, see `LICENSE`. Llama 2 is subject to Meta's Llama 2 Community License.
