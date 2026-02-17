from __future__ import annotations 
from .chunking import chunk_text 
from .embedder import OllamaEmbedder 
from .vectordb import VectorDB

class RAGPipeline: 
    def __init__(self): 
        self.embedder = OllamaEmbedder(model="nomic-embed-text")
        self.vdb = VectorDB(persist_dir=".chroma", collection_name="docs")

    def ingest_text(self, text: str, source: str, chunk_size: int = 1000, overlap: int = 200): 
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks: 
            return {"chunks_added": 0}
        
        texts = [c.text for c in chunks]
        ids = [f"{source}:{c.chunk_id}" for c in chunks]
        metadatas = [{"source": source, "start": c.start, "end":c.end} for c in chunks]

        embeddings = self.embedder.embed_texts(texts) 
        self.vdb.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        return {
            "chunks_added": len(chunks), 
            "chunk_size": chunk_size, 
            "overlap": overlap
        }
    def retrieve(self, question: str, top_k: int = 5): 
        q_emb = self.embedder.embed_query(question)
        res = self.vdb.query(query_embedding=q_emb, top_k=top_k)

        docs = res.get("documents", [[]])[0] 
        metas = res.get("metadatas", [[]])[0] 
        dists = res.get("distances", [[]])[0] 

        retrieved = [] 
        for doc, meta, dist in zip(docs, metas, dists): 
            retrieved.append({
                "text": doc, 
                "source": meta.get("source"), 
                "span": [meta.get("start"), meta.get("end")], 
                "distance": dist, 
            })

        return retrieved 
    
    def answer_grounded(self, question: str, top_k: int = 5): 
        retrieved = self.retrieve(question, top_k=top_k) 
        if not retrieved: 
            return {"answer": "I don't know (no relevant context retrieved).", "citations": [], "retrieved": []}
        
        best = retrieved[0] 
        answer = (
            "Grounded answer from retrieved context:\n\n"
            + best["text"][:1200] 
        )
        citations = [{"source": best["source"], "span": best["span"]}]
        return {"answer": answer, "citations": citations, "retrieved": retrieved}