import faiss
import numpy as np
import pickle
from typing import List
from src.embed import embed


class VectorStore:
    def __init__(self, dim=384):

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

        self.texts = []
        self.is_trained = False

    # =====================================================
    # NORMALIZATION (CRITICAL FOR COSINE SIMILARITY)
    # =====================================================
    def _normalize(self, x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    # =====================================================
    # ADD DOCUMENTS
    # =====================================================
    def add(self, texts: List[str]):

        if not texts:
            return

        vectors = np.array(embed(texts), dtype=np.float32)
        vectors = self._normalize(vectors)

        self.index.add(vectors)
        self.texts.extend(texts)

        self.is_trained = True

    # =====================================================
    # SEARCH (FIXED + FILTERED)
    # =====================================================
    def search(self, query: str, top_k=5, threshold=0.35):

        if not self.is_trained or len(self.texts) == 0:
            return []

        q_vec = np.array(embed([query]), dtype=np.float32)
        q_vec = self._normalize(q_vec)

        scores, idx = self.index.search(q_vec, top_k * 3)

        results = []

        for i, score in zip(idx[0], scores[0]):

            # 🔥 FILTER BAD MATCHES
            if score < threshold:
                continue

            if i < len(self.texts):
                results.append(self.texts[i])

            if len(results) >= top_k:
                break

        return results

    # =====================================================
    # SAVE
    # =====================================================
    def save(self, path="vector_store.pkl"):

        with open(path, "wb") as f:
            pickle.dump({
                "texts": self.texts
            }, f)

    # =====================================================
    # LOAD (SAFE REBUILD INDEX)
    # =====================================================
    def load(self, path="vector_store.pkl"):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.texts = data.get("texts", [])

        if self.texts:
            vectors = np.array(embed(self.texts), dtype=np.float32)
            vectors = self._normalize(vectors)

            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vectors)

            self.is_trained = True