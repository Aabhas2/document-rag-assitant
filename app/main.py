from __future__ import annotations 
from fastapi import FastAPI, UploadFile, File 
from pydantic import BaseModel 

from app.rag.rag_pipeline import RAGPipeline 

app = FastAPI(title="Mini Document RAG Assistant (Ollama + Chroma)") 
rag = RAGPipeline() 

class AskRequest(BaseModel): 
    question: str 
    top_k: int = 5 

@app.get("/health")
def health(): 
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), chunk_size: int = 1000, overlap: int = 200): 
    raw = await file.read() 
    text = raw.decode("utf-8", errors="ignore")
    result = rag.ingest_text(text=text, source=file.filename, chunk_size=chunk_size, overlap=overlap)
    return {"source": file.filename, **result}

@app.post("/ask")
def ask(req: AskRequest): 
    return rag.answer_grounded(req.question, top_k=req.top_k)