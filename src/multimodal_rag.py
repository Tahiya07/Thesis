import numpy as np
import faiss

from src.chunker import chunk_text
from src.embed import embed


class MultiModalRAG:

    def __init__(self):
        self.chunks = []
        self.sources = []
        self.embeddings = None
        self.index = None

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    def _normalize(self, x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    # -----------------------------
    # ADD TEXT (CORE)
    # -----------------------------
    def add_text(self, text: str, source: str = "unknown"):

        if not text or len(text.strip()) < 30:
            return

        chunks = chunk_text(text)

        chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

        if not chunks:
            return

        emb = np.array(embed(chunks), dtype=np.float32)
        emb = self._normalize(emb)

        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self._build_index()

        print(f"✅ Added {len(chunks)} chunks from {source}")

    # -----------------------------
    # PDF
    # -----------------------------
    def add_pdf(self, text: str):
        self.add_text(text, source="pdf")

    # -----------------------------
    # IMAGE (IMPORTANT FIX)
    # -----------------------------
    def add_image(self, image_text: str):
        self.add_text(image_text, source="image")

    # -----------------------------
    # INDEX
    # -----------------------------
    def _build_index(self):

        if self.embeddings is None or len(self.embeddings) == 0:
            return

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))

    # -----------------------------
    # RETRIEVAL
    # -----------------------------
    def retrieve(self, query: str, k: int = 3):

        if self.index is None:
            return []

        q = np.array(embed([query]), dtype=np.float32)
        q = self._normalize(q)

        scores, idxs = self.index.search(q, k * 5)

        results = []

        for i, idx in enumerate(idxs[0]):

            if idx >= len(self.chunks):
                continue

            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(scores[0][i])
            })

            if len(results) >= k:
                break

        return results