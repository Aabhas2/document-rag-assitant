from __future__ import annotations
import chromadb
from chromadb.config import Settings


class VectorDB:
    def __init__(self, persist_dir: str = ".chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    def get_collection(self, name: str):
        """Return (and create if needed) a Chroma collection by name."""
        return self.client.get_or_create_collection(name=name)

    def add(
        self,
        collection_name: str,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        col = self.get_collection(collection_name)
        col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ):
        col = self.get_collection(collection_name)
        return col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def reset(self, collection_name: str):
        """Delete and recreate a collection (clears all data for that namespace)."""
        try:
            self.client.delete_collection(name=collection_name)
        except ValueError:
            pass  # collection didn't exist yet — that's fine
        self.client.get_or_create_collection(name=collection_name)