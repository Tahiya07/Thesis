# src/rag_engine.py

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
# SIMPLE BM25 (NO EXTERNAL DEPENDENCY)
# =========================================================
class BM25:
    def __init__(self):
        self.docs = []
        self.doc_freq = {}
        self.avg_len = 0

    def add(self, docs):
        self.docs.extend(docs)
        self._recalc()

    def _recalc(self):
        self.doc_freq = {}
        total_len = 0

        for doc in self.docs:
            words = doc.lower().split()
            total_len += len(words)

            for w in set(words):
                self.doc_freq[w] = self.doc_freq.get(w, 0) + 1

        self.avg_len = total_len / (len(self.docs) + 1e-6)

    def score(self, query, doc):
        q_words = query.lower().split()
        d_words = doc.lower().split()

        score = 0
        for w in q_words:
            if w in d_words:
                idf = np.log((len(self.docs) + 1) / (1 + self.doc_freq.get(w, 0)))
                score += idf

        return score


# =========================================================
# PRODUCTION RAG ENGINE
# =========================================================
class RAGEngine:
    def __init__(self, persist_path="rag_store.pkl"):

        self.persist_path = persist_path

        # storage
        self.chunks = []
        self.embeddings = None

        # FAISS
        self.index = None

        # BM25
        self.bm25 = BM25()

        # metadata
        self.meta = []

        # evaluation stats
        self.stats = {
            "queries": 0,
            "retrieval_hits": 0,
            "avg_latency": 0.0
        }

        # load existing index if available
        self._load()

    # =====================================================
    # INGESTION
    # =====================================================
    def add_pdf(self, path):
        text = load_pdf_text(path)
        self._add_text(text, source="pdf")

    def add_url(self, url):
        text = load_webpage(url)
        self._add_text(text, source="url")

    def add_text(self, text, source="manual"):
        self._add_text(text, source=source)

    # =====================================================
    # CORE INGESTION PIPELINE
    # =====================================================
    def _add_text(self, text, source="unknown"):

        chunks = chunk_text(text)

        if not chunks:
            return

        # metadata tracking
        self.chunks.extend(chunks)
        self.meta.extend([source] * len(chunks))

        # BM25 update
        self.bm25.add(chunks)

        # embeddings (lazy incremental)
        new_emb = embed(chunks)
        new_emb = np.array(new_emb).astype("float32")

        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])

        # rebuild index
        self._build_index()

        # persist
        self._save()

    # =====================================================
    # INDEXING
    # =====================================================
    def _build_index(self):

        if self.embeddings is None or len(self.chunks) == 0:
            self.index = None
            return

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    # =====================================================
    # HYBRID RETRIEVAL (BM25 + VECTOR)
    # =====================================================
    def retrieve(self, query, k=5):

        if self.index is None:
            return []

        start = time.time()

        # vector search
        q_vec = embed([query]).astype("float32")
        scores, idxs = self.index.search(q_vec, k * 3)

        results = []

        for score, i in zip(scores[0], idxs[0]):
            if i >= len(self.chunks):
                continue

            chunk = self.chunks[i]

            bm25_score = self.bm25.score(query, chunk)

            # hybrid scoring
            final_score = (0.7 * float(score)) + (0.3 * bm25_score)

            results.append((final_score, chunk))

        results.sort(reverse=True, key=lambda x: x[0])

        latency = time.time() - start

        # stats update
        self.stats["queries"] += 1
        self.stats["avg_latency"] = (
            (self.stats["avg_latency"] * (self.stats["queries"] - 1) + latency)
            / self.stats["queries"]
        )

        if len(results) > 0:
            self.stats["retrieval_hits"] += 1

        # safety cap to avoid overload
        k = min(k, 4)
        return [r[1] for r in results[:k]]

    # =====================================================
    # EVALUATION METRICS
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