# Mini Document RAG Assistant (FastAPI + Chroma + Ollama Embeddings)

End-to-end RAG demo:
- Upload documents -> chunk -> embed (Ollama) -> store (Chroma)
- Ask questions -> retrieve top-k -> grounded answer with citations
- Includes a small evaluation script measuring retrieval hit-rate

## Tech
FastAPI, ChromaDB, Ollama (nomic-embed-text), Python

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
