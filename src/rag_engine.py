import numpy as np
import faiss
import random

from src.chunker import chunk_text
from src.embed import embed
from src.privacy.privacy_model import PrivacyClassifier
from src.privacy.privacy import is_private
from src.ldl.bloom_ldl import BloomLDL


class RAGEngine:

    def __init__(self, privacy_model=None, seed=42):

        # =========================
        # REPRODUCIBILITY
        # =========================
        np.random.seed(seed)
        random.seed(seed)

        self.chunks = []
        self.embeddings = None
        self.index = None

        # =========================
        # PRIVACY MODULES
        # =========================
        self.privacy_model = privacy_model
        self.rule_model = PrivacyClassifier()

        self.use_learning_privacy = privacy_model is not None
        self.use_privacy = True
        self.use_diversity = True

        # =========================
        # LDL (BLOOM TAXONOMY MODULE)
        # =========================
        self.ldl = BloomLDL()
        self.use_ldl = True

        # =========================
        # THESIS PARAMETER
        # =========================
        self.lambda_privacy = 0.3

    # =====================================================
    # NORMALIZATION
    # =====================================================
    def _normalize(self, x):
        if x is None or len(x) == 0:
            return x
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norm

    # =====================================================
    # PRIVACY SCORE
    # =====================================================
    def _privacy_score(self, text: str) -> float:

        rule_score = 1.0 if is_private(text) else 0.0

        if self.use_learning_privacy and self.privacy_model is not None:
            learned_score = float(self.privacy_model.predict_proba(text))
            return float((rule_score + learned_score) / 2.0)

        return rule_score

    # =====================================================
    # ADD TEXT
    # =====================================================
    def add_text(self, text: str):

        if not text or len(text.strip()) < 150:
            return

        chunks = chunk_text(text)

        seen, filtered = set(), []

        for c in chunks:
            c = c.strip()

            if len(c) < 40:
                continue

            if is_private(c):
                continue

            key = c[:120]
            if key in seen:
                continue

            seen.add(key)
            filtered.append(c)

        if not filtered:
            return

        emb = np.array(embed(filtered), dtype=np.float32)

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        emb = self._normalize(emb)

        self.chunks.extend(filtered)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self._build_index()

    # =====================================================
    # FAISS INDEX
    # =====================================================
    def _build_index(self):

        if self.embeddings is None or len(self.embeddings) == 0:
            return

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))

    # =====================================================
    # COSINE DIVERSITY
    # =====================================================
    def _cosine(self, a, b):
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)

    # =====================================================
    # LDL INFERENCE (NEW - THESIS CORE FIX)
    # =====================================================
    def _ldl_predict(self, text: str):

        if not self.use_ldl:
            return {
                "bloom_level": None,
                "uncertainty": 0.0
            }

        pred = self.ldl.predict(text)

        if isinstance(pred, tuple) and len(pred) == 2:
            bloom_level, dist = pred
            uncertainty = self.ldl.uncertainty(dist) if hasattr(self.ldl, "uncertainty") else 0.0
            return {
                "bloom_level": bloom_level,
                "uncertainty": float(uncertainty)
            }

        if isinstance(pred, dict):
            return {
                "bloom_level": pred.get("bloom_level") or pred.get("level"),
                "uncertainty": float(pred.get("uncertainty", 0.0))
            }

        return {
            "bloom_level": None,
            "uncertainty": 0.0
        }

    # =====================================================
    # RETRIEVAL (FINAL THESIS VERSION)
    # =====================================================
    def retrieve(self, query, k=3):

        if self.index is None or len(self.chunks) == 0:
            return []

        q = np.array(embed([query]), dtype=np.float32)

        if q.ndim == 1:
            q = q.reshape(1, -1)

        q = self._normalize(q)

        scores, idxs = self.index.search(q, k * 10)

        results = []
        selected_embeddings = []

        query_sensitive = is_private(query)

        for i, idx in enumerate(idxs[0]):

            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            chunk_emb = self.embeddings[idx]

            # =========================
            # SEMANTIC SCORE
            # =========================
            semantic = float(scores[0][i])
            semantic = max(0.0, min(1.0, semantic))

            # =========================
            # PRIVACY SCORE
            # =========================
            privacy = self._privacy_score(chunk)

            if query_sensitive:
                privacy *= 1.2

            final_score = max(0.0, semantic - self.lambda_privacy * privacy)

            if self.use_privacy and final_score < 0.30:
                continue

            # =========================
            # DIVERSITY
            # =========================
            if self.use_diversity:
                if any(self._cosine(chunk_emb, e) > 0.85 for e in selected_embeddings):
                    continue
                selected_embeddings.append(chunk_emb)

            # =========================
            # LDL OUTPUT (CRITICAL FIX)
            # =========================
            ldl_out = self._ldl_predict(chunk)

            results.append({
                "text": chunk,
                "score": float(final_score),
                "semantic_score": float(semantic),
                "privacy_score": float(privacy),
                "raw_faiss_score": float(scores[0][i]),

                # ✅ LDL integration (NOW THESIS-CORRECT)
                "bloom_level": ldl_out.get("bloom_level"),
                "uncertainty": ldl_out.get("uncertainty", 0.0),
            })

            if len(results) >= k:
                break

        return results

    # =====================================================
    # ABLATION CONTROL
    # =====================================================
    def set_ablation(
        self,
        use_privacy=None,
        use_diversity=None,
        use_learning_privacy=None,
        lambda_privacy=None
    ):

        if use_privacy is not None:
            self.use_privacy = use_privacy

        if use_diversity is not None:
            self.use_diversity = use_diversity

        if use_learning_privacy is not None:
            self.use_learning_privacy = use_learning_privacy

        if lambda_privacy is not None:
            self.lambda_privacy = float(lambda_privacy)
