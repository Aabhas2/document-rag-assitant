from __future__ import annotations 
from dataclasses import dataclass 


@dataclass 
class Chunk: 
    text: str 
    start: int 
    end: int 
    chunk_id: str  

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    # Character-based chunking with overlap 
    text = text.replace("\r\n", "\n").strip() 
    if not text: 
        return [] 
    
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = [] 
    i = 0 
    idx = 0 

    while i < len(text): 
        start = i 
        end = min(len(text), i + chunk_size)
        chunk = text[start:end].strip() 
        if chunk: 
            chunks.append(Chunk(text=chunk, start=start, end=end, chunk_id=f"chunk-{idx:04d}"))
            idx += 1
        i += step 

    return chunks 