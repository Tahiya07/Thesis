import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CORE EVALUATION ENGINE (THESIS-GRADE)
# =========================================================
class RAGEvaluator:

    # -------------------------
    # LATENCY
    # -------------------------
    def measure_latency(self, func, *args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        return output, end - start

    # -------------------------
    # BLEU SCORE
    # -------------------------
    def compute_bleu(self, reference, candidate):
        if not reference or not candidate:
            return 0.0

        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        smoothie = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

    # -------------------------
    # TF-IDF SIMILARITY (HALLUCINATION PROXY)
    # -------------------------
    def compute_context_similarity(self, context, answer):
        if not context or not answer:
            return 0.0

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([context, answer])

        return float(cosine_similarity(vectors[0], vectors[1])[0][0])

    # -------------------------
    # HALLUCINATION SCORE (INVERTED SIMILARITY)
    # -------------------------
    def hallucination_score(self, context, answer):
        return 1.0 - self.compute_context_similarity(context, answer)

    # -------------------------
    # FAITHFULNESS
    # -------------------------
    def faithfulness(self, context, answer):
        return self.compute_context_similarity(context, answer)

    # -------------------------
    # LENGTH METRICS
    # -------------------------
    def answer_length_metrics(self, answer):
        if not answer:
            return {"char_len": 0, "word_len": 0}

        return {
            "char_len": len(answer),
            "word_len": len(answer.split())
        }

    # -------------------------
    # PRECISION@K (RAG QUALITY)
    # -------------------------
    def precision_at_k(self, retrieved, relevant_keywords):
        if not retrieved:
            return 0.0

        hits = 0
        for r in retrieved:
            if any(k.lower() in r.lower() for k in relevant_keywords):
                hits += 1

        return hits / len(retrieved)

    # -------------------------
    # RECALL@K
    # -------------------------
    def recall_at_k(self, retrieved, relevant_keywords):
        if not relevant_keywords:
            return 0.0

        hits = 0
        for k in relevant_keywords:
            if any(k.lower() in r.lower() for r in retrieved):
                hits += 1

        return hits / len(relevant_keywords)

    # -------------------------
    # LLM-BASED JUDGE (UNCHANGED BUT SAFE)
    # -------------------------
    def llm_hallucination_judge(self, llm, context, answer):

        prompt = f"""
You are an evaluation system.

Check if the ANSWER is supported by CONTEXT.

Return ONLY a number between 0 and 1:
1.0 = fully supported
0.5 = partially supported
0.0 = not supported

CONTEXT:
{context}

ANSWER:
{answer}

Score:
"""

        try:
            from src.llm import generate

            result = generate(
                llm,
                prompt=prompt,
                temperature=0.0,
                max_tokens=5
            )

            score = float(result["response"].strip())
            return max(0.0, min(1.0, score))

        except:
            return 0.0