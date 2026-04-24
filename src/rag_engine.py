import os
import time
import pickle
import numpy as np
import faiss

from src.chunker import chunk_text
from src.embed import embed
from src.loaders.pdf_loader import load_pdf_text
from src.loaders.web_loader import load_webpage


# =========================================================
# BM25 (FIXED LIGHTWEIGHT VERSION)
# =========================================================
class BM25:
    def __init__(self):
        self.docs = []
        self.doc_freq = {}

    def add(self, docs):
        self.docs.extend(docs)
        self._recalc()

    def _recalc(self):
        self.doc_freq = {}

        for doc in self.docs:
            words = doc.lower().split()
            unique_words = set(words)

            for w in unique_words:
                self.doc_freq[w] = self.doc_freq.get(w, 0) + 1

    def score(self, query, doc):
        q_words = query.lower().split()
        d_words = doc.lower().split()

        doc_len = len(d_words)
        score = 0.0

        for w in q_words:
            if w in d_words:
                tf = d_words.count(w)
                idf = np.log((len(self.docs) + 1) / (1 + self.doc_freq.get(w, 0)))
                score += (tf * idf) / (doc_len + 1e-8)

        return score


# =========================================================
# RAG ENGINE (FIXED + STABLE HYBRID RETRIEVAL)
# =========================================================
class RAGEngine:
    def __init__(self, persist_path="rag_store.pkl"):

        self.persist_path = persist_path

        self.chunks = []
        self.embeddings = None
        self.index = None
        self.meta = []

        self.bm25 = BM25()

        self.stats = {
            "queries": 0,
            "retrieval_hits": 0,
            "avg_latency": 0.0,
            "total_latency": 0.0
        }

        self._load()

    # =====================================================
    # INGESTION
    # =====================================================
    def add_pdf(self, path):
        text = load_pdf_text(path)
        self._add_text(text, "pdf")

    def add_url(self, url):
        text = load_webpage(url)
        self._add_text(text, "url")

    def add_text(self, text, source="manual"):
        self._add_text(text, source)

    def _add_text(self, text, source="unknown"):

        chunks = chunk_text(text)
        if not chunks:
            return

        self.chunks.extend(chunks)
        self.meta.extend([source] * len(chunks))

        self.bm25.add(chunks)

        new_emb = np.array(embed(chunks)).astype("float32")

        # =========================
        # FIX 1: NORMALIZE EMBEDDINGS (CRITICAL)
        # =========================
        new_emb = new_emb / (np.linalg.norm(new_emb, axis=1, keepdims=True) + 1e-8)

        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])

        self._build_index()
        self._save()

    # =====================================================
    # INDEX BUILDING
    # =====================================================
    def _build_index(self):

        if self.embeddings is None or len(self.chunks) == 0:
            self.index = None
            return

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    # =====================================================
    # RETRIEVAL (FIXED HYBRID SCORING)
    # =====================================================
    def retrieve(self, query, k=5):

        if self.index is None:
            return []

        start = time.time()

        q_vec = np.array(embed([query])).astype("float32")

        # =========================
        # FIX 2: NORMALIZE QUERY EMBEDDING
        # =========================
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-8)

        scores, idxs = self.index.search(q_vec, k * 3)

        results = []

        for i, idx in enumerate(idxs[0]):
            if idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            faiss_score = float(scores[0][i])
            bm25_score = self.bm25.score(query, chunk)

            results.append((faiss_score, bm25_score, chunk))

        if not results:
            return []

        # =========================
        # FIX 3: NORMALIZE SCORES BEFORE FUSION
        # =========================
        faiss_scores = np.array([r[0] for r in results])
        bm25_scores = np.array([r[1] for r in results])

        faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.ptp() + 1e-8)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)

        final_results = []

        for i in range(len(results)):
            score = 0.7 * faiss_scores[i] + 0.3 * bm25_scores[i]
            final_results.append((score, results[i][2]))

        final_results.sort(reverse=True, key=lambda x: x[0])

        latency = time.time() - start

        # =========================
        # STATS
        # =========================
        self.stats["queries"] += 1
        self.stats["total_latency"] += latency
        self.stats["avg_latency"] = self.stats["total_latency"] / self.stats["queries"]

        if len(final_results) > 0 and final_results[0][0] > 0.3:
            self.stats["retrieval_hits"] += 1

        return [r[1] for r in final_results[:k]]

    # =====================================================
    # STATS
    # =====================================================
    def get_stats(self):
        return {
            "total_queries": self.stats["queries"],
            "hit_rate": self.stats["retrieval_hits"] / max(1, self.stats["queries"]),
            "avg_latency_sec": self.stats["avg_latency"],
            "chunks": len(self.chunks)
        }

    # =====================================================
    # PERSISTENCE
    # =====================================================
    def _save(self):
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                    "meta": self.meta,
                    "stats": self.stats
                }, f)
        except:
            pass

    def _load(self):
        if not os.path.exists(self.persist_path):
            return

        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)

            self.chunks = data.get("chunks", [])
            self.embeddings = data.get("embeddings", None)
            self.meta = data.get("meta", [])
            self.stats = data.get("stats", self.stats)

            if self.embeddings is not None:
                self._build_index()

            self.bm25.add(self.chunks)

        except:
            pass