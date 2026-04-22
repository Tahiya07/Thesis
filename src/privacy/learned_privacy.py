import numpy as np
import faiss
import random

from src.embed import embed
from src.chunker import chunk_text
from src.loaders.pdf_loader import load_pdf_text
from src.image_pipeline import ImagePipeline
from src.loaders.web_loader import load_webpage
from src.blip_captioner import BLIPCaptioner


# =========================================================
# PRIVACY MODULE
# =========================================================
class PrivacySecurityModel:
    def __init__(self):
        self.private_keywords = [
            "student id", "password", "exam",
            "marks", "email", "phone", "address"
        ]

    def privacy_risk(self, text: str) -> float:
        t = text.lower()
        hits = sum(k in t for k in self.private_keywords)
        return min(1.0, hits / 3.0)

    def leakage_risk(self, query: str, text: str) -> float:
        q = query.lower()
        t = text.lower()

        trigger_words = ["show", "extract", "list", "give", "who", "what"]
        trigger_score = sum(w in q for w in trigger_words)

        overlap = len(set(q.split()) & set(t.split())) / (len(t.split()) + 1e-6)

        return min(1.0, trigger_score * overlap)


# =========================================================
# RAG ENGINE (STABLE VERSION)
# =========================================================
class RAGEngine:
    def __init__(self, lambda_privacy=0.4, beta_leakage=0.3):

        self.text_chunks = []
        self.image_chunks = []

        self.text_vectors = []
        self.image_vectors = []

        self.index = None
        self.all_chunks = []

        self.clip = BLIPCaptioner()
        self.security = PrivacySecurityModel()

        self.lambda_privacy = lambda_privacy
        self.beta_leakage = beta_leakage

    # -------------------------
    # INGESTION
    # -------------------------
    def build_from_pdf(self, path: str):
        text = load_pdf_text(path)
        self._add_text(text)

    def build_from_image(self, path: str):
        pipeline = ImagePipeline()

        result = pipeline.process(path)

        # store fused representation only
        self._add_text(result["fused_text"])

    def build_from_text(self, text: str):
        self._add_text(text)

    def build_from_url(self, url: str):
        text = load_webpage(url)
        self._add_text(text)

    # -------------------------
    # INTERNAL ADD
    # -------------------------
    from src.summarizer import summarize

    def _add_text(self, text: str):
        """
        1. Chunk text
        2. Summarize each chunk
        3. Embed summary (NOT raw text)
        """

        chunks = chunk_text(text)

        processed_chunks = []

        for c in chunks:
            if not c or not c.strip():
                continue

            try:
                # =========================
                # SUMMARIZATION STEP
                # =========================
                summary = summarize(self.llm if hasattr(self, "llm") else None, c)

                # fallback if LLM not available in this context
                if not summary:
                    summary = c

            except:
                summary = c

            processed_chunks.append(summary)

        # store summaries (important for RAG efficiency)
        self.text_chunks.extend(processed_chunks)

        # embed summaries
        if len(processed_chunks) > 0:
            self.text_vectors = embed(processed_chunks)

    def add_image(self, path):
        caption = self.clip.caption(path)
        self.rag._add_text(caption)
        self.rag.build_index()
    # -------------------------
    # INDEX BUILD (SAFE)
    # -------------------------
    def _build_index(self):

        self.all_chunks = self.text_chunks + self.image_chunks

        # SAFE FALLBACK
        if len(self.all_chunks) == 0:
            self.index = None
            return

        vectors = []

        if len(self.text_vectors) > 0:
            vectors.extend(self.text_vectors)

        if len(vectors) == 0:
            self.index = None
            return

        vectors = np.array(vectors).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    # -------------------------
    # RETRIEVAL (SAFE)
    # -------------------------
    def retrieve(self, query: str, k: int = 5):

        if self.index is None:
            self._build_index()

        if self.index is None:
            return []  # IMPORTANT: clean fallback

        q_vec = embed([query]).astype("float32")

        scores, idxs = self.index.search(q_vec, k * 3)

        results = []

        for score, i in zip(scores[0], idxs[0]):
            if i >= len(self.all_chunks):
                continue

            chunk = self.all_chunks[i]

            p_risk = self.security.privacy_risk(chunk)
            l_risk = self.security.leakage_risk(query, chunk)

            final_score = score - self.lambda_privacy * p_risk - self.beta_leakage * l_risk

            results.append((final_score, chunk))

        results.sort(reverse=True, key=lambda x: x[0])

        return [r[1] for r in results[:k]]

    # -------------------------
    # MAIN ASK FUNCTION (DUAL MODE)
    # -------------------------
    def ask(self, llm, question: str, bloom_classifier=None):

        # BLOOM CLASSIFICATION
        try:
            bloom = bloom_classifier(question) if bloom_classifier else "Understand"
        except:
            bloom = "Understand"

        # RETRIEVAL
        context = self.retrieve(question)
        use_rag = len(context) > 0

        context_text = "\n".join(context) if use_rag else ""

        # PROMPT
        if use_rag:
            prompt = f"""
You are a privacy-preserving academic assistant.

Mode: RAG
Bloom Level: {bloom}

Context:
{context_text}

Question:
{question}

Answer:
"""
        else:
            prompt = f"""
You are an academic assistant.

Mode: LLM-only
Bloom Level: {bloom}

Question:
{question}

Answer:
"""

        # LLM CALL
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Academic assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )

        return {
            "answer": response["choices"][0]["message"]["content"],
            "bloom_level": bloom,
            "used_rag": use_rag
        }