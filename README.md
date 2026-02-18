# Mini Document RAG Assistant

A local, privacy-first Retrieval-Augmented Generation assistant. Upload `.txt` documents, ask natural-language questions, and get grounded answers with source citations — all running on your machine with no cloud APIs.

https://github.com/user-attachments/assets/83f54497-b39c-4f0d-92d2-569b3c53537c

## Features

- **Document ingestion** — upload text files that are chunked, embedded, and stored in a persistent vector database.
- **Namespace isolation** — each workspace (e.g. `movies`, `mit`) gets its own Chroma collection so answers never leak across topics.
- **Grounded answers** — top-3 retrieved chunks are returned as context with source + character-span citations.
- **Reset workspace** — clear a single namespace without touching others.
- **Evaluation script** — measure retrieval hit-rate against a question set.

## Tech Stack

| Layer | Tool |
|-------|------|
| API | FastAPI + Uvicorn |
| Embeddings | Ollama (`nomic-embed-text`) — local, via HTTP |
| Vector store | ChromaDB (persistent) |
| Frontend | Streamlit |
| Language | Python 3.12 |

## Project Structure

```
app/
  main.py              # FastAPI endpoints (/ingest, /ask, /reset, /health)
  rag/
    chunking.py         # Character-based chunking with overlap
    embedder.py         # OllamaEmbedder (calls localhost:11434)
    rag_pipeline.py     # Ingest → retrieve → answer pipeline
    vectordb.py         # ChromaDB wrapper with per-namespace collections
ui.py                   # Streamlit frontend
scripts/
  eval.py               # Retrieval hit-rate evaluation
  eval_questions.json   # Question set for eval
data/sample_docs/       # Sample .txt files (alice.txt, mit_license.txt)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest?namespace=&chunk_size=&overlap=` | Upload & ingest a `.txt` file |
| `POST` | `/ask` | Ask a question `{"question", "namespace", "top_k"}` |
| `POST` | `/reset?namespace=` | Delete & recreate a namespace collection |

## Getting Started

### Prerequisites

- **Python 3.12+**
- **Ollama** installed and running with the `nomic-embed-text` model:
  ```bash
  ollama pull nomic-embed-text
  ollama serve            # runs on localhost:11434
  ```

### Install & Run

```bash
git clone https://github.com/<your-username>/document-rag-assistant.git
cd document-rag-assistant

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload --port 8000

# In a second terminal — start the UI
streamlit run ui.py
```

Open **http://localhost:8501** in your browser. Set a namespace, upload a file, and ask questions.

## Quick Test (cURL)

```bash
# Reset a workspace
curl -X POST "http://127.0.0.1:8000/reset?namespace=mit"

# Ingest a document
curl -X POST "http://127.0.0.1:8000/ingest?namespace=mit" \
  -F file=@data/sample_docs/mit_license.txt

# Ask a question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the MIT license allow?","namespace":"mit","top_k":5}'
```

## License

MIT
