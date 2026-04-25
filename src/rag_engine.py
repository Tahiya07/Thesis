import numpy as np
import faiss

from src.chunker import chunk_text
from src.embed import embed
from src.privacy.privacy_filter import PrivacyFilter
from src.privacy.privacy import is_private


class RAGEngine:

    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.privacy_filter = PrivacyFilter()

    # =====================================================
    def _normalize(self, x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    # =====================================================
    # ADD TEXT
    # =====================================================
    def add_text(self, text: str):

        if not text or len(text.strip()) < 150:
            print("⚠️ Skipping small text")
            return

        chunks = chunk_text(text)

        seen = set()
        filtered = []

        for c in chunks:
            c = c.strip()

            if len(c) < 40:
                continue

            key = c[:100]

            if key in seen:
                continue

            seen.add(key)
            filtered.append(c)

        if not filtered:
            print("⚠️ No valid chunks after filtering")
            return

        emb = np.array(embed(filtered), dtype=np.float32)
        emb = self._normalize(emb)

        self.chunks.extend(filtered)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self._build_index()

        print(f"✅ Indexed {len(filtered)} chunks")

    # =====================================================
    def _build_index(self):

        if self.embeddings is None or len(self.embeddings) == 0:
            return

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))

    # =====================================================
    # LIGHTWEIGHT SEMANTIC SIMILARITY
    # =====================================================
    def _cosine(self, a, b):
        return np.dot(a, b) / (
            (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        )

    # =====================================================
    # RETRIEVE (FINAL OPTIMIZED)
    # =====================================================
    def retrieve(self, query, k=3, lambda_privacy=0.3):

        if self.index is None:
            return []

        q = np.array(embed([query]), dtype=np.float32)
        q = self._normalize(q)

        scores, idxs = self.index.search(q, k * 10)

        results = []
        selected_embeddings = []

        query_sensitive = is_private(query)

        for i, idx in enumerate(idxs[0]):

            if idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            chunk_emb = self.embeddings[idx]

            # -----------------------------
            # SEMANTIC SCORE (0–1)
            # -----------------------------
            semantic = float(scores[0][i])
            semantic = max(0.0, min(1.0, (semantic + 1) / 2))

            # -----------------------------
            # PRIVACY PENALTY
            # -----------------------------
            privacy = self.privacy_filter.risk_score(chunk)

            if query_sensitive:
                privacy *= 1.2

            final_score = semantic - lambda_privacy * privacy

            if final_score < 0.30:
                continue

            # -----------------------------
            # 🔥 SEMANTIC DEDUP (KEY UPGRADE)
            # -----------------------------
            is_duplicate = False

            for prev_emb in selected_embeddings:
                sim = self._cosine(chunk_emb, prev_emb)

                if sim > 0.85:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            selected_embeddings.append(chunk_emb)

            results.append({
                "text": chunk,
                "score": float(final_score)
            })

            if len(results) >= k:
                break

        return results