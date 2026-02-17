from __future__ import annotations 
import chromadb 
from chromadb.config import Settings 


class VectorDB: 
    def __init__(self, persist_dir: str = ".chroma", collection_name: str = "docs"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, ids: list[str], texts: list[str],embeddings: list[list[float]], metadatas: list[dict]):
        self.collection.add(
            ids=ids, 
            documents=texts, 
            embeddings=embeddings, 
            metadatas=metadatas
        )

    def query(self, query_embedding: list[float], top_k: int = 5): 
        return self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k, 
            include=['documents','metadatas','distances']
        )
    
    