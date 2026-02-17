from __future__ import annotations 
import requests 

class OllamaEmbedder: 
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://127.0.0.1:11434"):
        self.model = model 
        self.base_url = base_url.rstrip("/")

    def embed_texts(self, texts: list[str]) -> list[list[float]]: 
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model":self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status() 
        data = r.json() 
        return data["embeddings"]
    
    def embed_query(self,query: str) -> list[float]: 
        return self.embed_texts([query])[0]