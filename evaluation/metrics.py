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
    # BLEU SCORE
    # =====================================================
    def compute_bleu(self, reference, candidate):
        if not reference or not candidate:
            return 0.0

        return sentence_bleu(
            [reference.split()],
            candidate.split(),
            smoothing_function=SmoothingFunction().method1
        )

    # =====================================================
    # SEMANTIC SIMILARITY (FAITHFULNESS CORE)
    # =====================================================
    def _similarity(self, context, answer):
        if not context or not answer:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            vecs = vectorizer.fit_transform([context, answer])
            return float(cosine_similarity(vecs[0], vecs[1])[0][0])
        except:
            return 0.0

    def faithfulness(self, context, answer):
        return self._similarity(context, answer)

    def hallucination_score(self, context, answer):
        return 1.0 - self._similarity(context, answer)

    # =====================================================
    # PRIVACY LEAKAGE (SAFE THESIS VERSION)
    # =====================================================
    def privacy_leakage(self, answer):
        sensitive = ["student id", "password", "marks", "grade", "exam"]
        answer = (answer or "").lower()

        hits = sum(1 for k in sensitive if k in answer)
        return hits / len(sensitive)

    # =====================================================
    # LENGTH METRICS
    # =====================================================
    def answer_length_metrics(self, answer):
        return {
            "char_len": len(answer or ""),
            "word_len": len((answer or "").split())
        }

    # =====================================================
    # RETRIEVAL PRECISION@K (FIXED LOGIC)
    # =====================================================
    def precision_at_k(self, retrieved_texts, keywords):
        if not retrieved_texts or not keywords:
            return 0.0

        retrieved_texts = [r.lower() for r in retrieved_texts]
        keywords = [k.lower() for k in keywords]

        hits = 0

        for r in retrieved_texts:
            if any(k in r for k in keywords):
                hits += 1

        return hits / len(retrieved_texts)

    # =====================================================
    # UNCERTAINTY SCORE
    # =====================================================
    def uncertainty_score(self, flag):
        return 1.0 if flag else 0.0

    # =====================================================
    # MAIN EVALUATION PIPELINE
    # =====================================================
    def evaluate_sample(self, system, question, reference=None, keywords=None):

        result, latency = self.measure_latency(system.ask, question)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])

        context = "\n".join([c["text"] for c in chunks]) if chunks else ""

        return {
            "question": question,
            "answer": answer,
            "latency": latency,

            "bleu": self.compute_bleu(reference, answer) if reference else 0.0,

            "faithfulness": self.faithfulness(context, answer),
            "hallucination": self.hallucination_score(context, answer),

            "privacy_leakage": self.privacy_leakage(answer),

            "uncertainty": self.uncertainty_score(result.get("uncertain", False)),

            "context_used": result.get("context_used", False),

            "length": self.answer_length_metrics(answer),

            # FIXED SAFE VERSION
            "precision@k": self.precision_at_k(
                [c["text"] for c in chunks],
                keywords or []
            ),
        }

    # =====================================================
    # BLOOM CONFIDENCE
    # =====================================================
    def bloom_confidence(self, dist):
        if dist is None or len(dist) == 0:
            return 0.0
        return float(np.max(dist))