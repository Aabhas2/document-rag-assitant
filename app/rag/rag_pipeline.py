from __future__ import annotations

import re
from typing import Optional

from .chunking import chunk_text
from .embedder import OllamaEmbedder
from .vectordb import VectorDB


class RAGPipeline:
    def __init__(self):
        self.embedder = OllamaEmbedder(model="nomic-embed-text")
        self.vdb = VectorDB(persist_dir=".chroma")

    # ── Ingest ───────────────────────────────────────────────
    def ingest_text(
        self,
        text: str,
        source: str,
        namespace: str = "default",
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {"chunks_added": 0}

        texts = [c.text for c in chunks]
        ids = [f"{source}:{c.chunk_id}" for c in chunks]
        metadatas = [{"source": source, "start": c.start, "end": c.end} for c in chunks]

        embeddings = self.embedder.embed_texts(texts)
        self.vdb.add(
            collection_name=namespace,
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {"chunks_added": len(chunks), "chunk_size": chunk_size, "overlap": overlap, "namespace": namespace}

    # ── Retrieve ─────────────────────────────────────────────
    def retrieve(self, question: str, namespace: str = "default", top_k: int = 5):
        q_emb = self.embedder.embed_query(question)
        res = self.vdb.query(collection_name=namespace, query_embedding=q_emb, top_k=top_k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        retrieved = []
        for doc, meta, dist in zip(docs, metas, dists):
            retrieved.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "span": [meta.get("start"), meta.get("end")],
                    "distance": dist,
                }
            )

        return retrieved

    # ── Smart deterministic helpers (Option A) ───────────────
    def _detect_startswith_letter_query(self, question: str) -> Optional[str]:
        """
        Detect queries like:
        - "Which words start with the letter 'N'?"
        - "words starting with N"
        Returns the target letter (uppercase) or None.
        """
        q = question.strip()

        # letter in quotes
        m = re.search(r"letter\s*['\"]([A-Za-z])['\"]", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # "starting with N" / "start with N"
        m = re.search(r"(start|starting)\s+with\s+['\"]?([A-Za-z])['\"]?\b", q, flags=re.IGNORECASE)
        if m:
            return m.group(2).upper()

        return None

    def _extract_words_starting_with(self, text: str, letter: str) -> list[str]:
        """
        Extract unique 'word-like' tokens starting with `letter`.
        Keeps simple tokens; strips punctuation.
        """
        # Pick up word tokens, including hyphen words like "Max-Planck"
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", text)
        out = []
        seen = set()
        for t in tokens:
            if t[:1].upper() == letter:
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(t)
        return out

    # ── Answer (grounded + smart mode) ───────────────────────
    def answer_grounded(self, question: str, namespace: str = "default", top_k: int = 5):
        retrieved = self.retrieve(question, namespace=namespace, top_k=top_k)
        if not retrieved:
            return {
                "answer": "I don't know (no relevant context retrieved).",
                "citations": [],
                "retrieved": [],
            }

        # Build context from top chunks (more stable than only top-1)
        top_chunks = retrieved[: min(3, len(retrieved))]
        context = "\n\n---\n\n".join(c["text"] for c in top_chunks)

        citations = [{"source": c["source"], "span": c["span"]} for c in top_chunks]

        # ✅ OPTION A: Detect deterministic "startswith letter" queries
        letter = self._detect_startswith_letter_query(question)
        if letter:
            words = self._extract_words_starting_with(context, letter)
            if not words:
                answer = f"No words starting with '{letter}' found in the retrieved context."
            else:
                answer = f"Words starting with '{letter}':\n- " + "\n- ".join(words)

            return {
                "answer": answer,
                "citations": citations,
                "retrieved": retrieved,
                "mode": "deterministic_startswith",
            }

        # Default MVP: show grounded context
        context_trimmed = context[:3000]
        answer = "Grounded answer from retrieved context:\n\n" + context_trimmed

        return {"answer": answer, "citations": citations, "retrieved": retrieved, "mode": "context_dump"}

    # ── Reset namespace ──────────────────────────────────────
    def reset(self, namespace: str = "default"):
        self.vdb.reset(collection_name=namespace)
        return {"reset": namespace}
