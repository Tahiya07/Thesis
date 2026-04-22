import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:

    # =====================================================
    # LATENCY
    # =====================================================
    def measure_latency(self, func, *args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        return out, time.time() - start

    # =====================================================
    # BLEU
    # =====================================================
    def compute_bleu(self, reference, candidate):
        if not reference or not candidate:
            return 0.0

        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        return sentence_bleu(
            [ref_tokens],
            cand_tokens,
            smoothing_function=SmoothingFunction().method1
        )

    # =====================================================
    # SIMILARITY (GROUNDING)
    # =====================================================
    def _similarity(self, context, answer):
        if not context or not answer:
            return 0.0

        vectorizer = TfidfVectorizer()
        vecs = vectorizer.fit_transform([context, answer])

        return float(cosine_similarity(vecs[0], vecs[1])[0][0])

    def hallucination_score(self, context, answer):
        return 1.0 - self._similarity(context, answer)

    def faithfulness(self, context, answer):
        return self._similarity(context, answer)

    # =====================================================
    # LENGTH
    # =====================================================
    def answer_length_metrics(self, answer):
        return {
            "char_len": len(answer or ""),
            "word_len": len((answer or "").split())
        }

    # =====================================================
    # RETRIEVAL METRICS
    # =====================================================
    def precision_at_k(self, retrieved, keywords):
        if not retrieved or not keywords:
            return 0.0

        hits = sum(
            any(k.lower() in r.lower() for k in keywords)
            for r in retrieved
        )
        return hits / len(retrieved)

    def recall_at_k(self, retrieved, keywords):
        if not retrieved or not keywords:
            return 0.0

        hits = sum(
            any(k.lower() in r.lower() for r in retrieved)
            for k in keywords
        )
        return hits / len(keywords)

    # =====================================================
    # FIXED LLM JUDGE (CRITICAL FIX)
    # =====================================================
    def llm_hallucination_judge(self, llm, context, answer):

        prompt = f"""
You are an evaluator.

Score how grounded the answer is in the context.

Return ONLY a number between 0 and 1.

Context:
{context}

Answer:
{answer}

Score:
"""

        try:
            from src.llm import generate

            res = generate(
                llm,
                prompt=prompt,
                temperature=0.0,
                max_tokens=5
            )

            return float(res["response"].strip())
        except:
            return 0.0