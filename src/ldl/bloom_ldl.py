import numpy as np
import re
from collections import Counter


class BloomLDL:

    def __init__(self):
        self.levels = [
            "Remember",
            "Understand",
            "Apply",
            "Analyze",
            "Evaluate",
            "Create"
        ]

        # keyword heuristics (CPU-friendly surrogate model)
        self.patterns = {
            "Remember": ["define", "list", "what is", "who is", "when"],
            "Understand": ["explain", "describe", "summarize"],
            "Apply": ["use", "solve", "calculate", "demonstrate"],
            "Analyze": ["compare", "analyze", "differentiate", "why"],
            "Evaluate": ["evaluate", "critique", "assess", "justify"],
            "Create": ["design", "propose", "develop", "formulate"]
        }

    # =====================================================
    # TEXT CLEANING
    # =====================================================
    def _clean(self, text):
        return (text or "").lower().strip()

    # =====================================================
    # SOFT MATCH SCORE
    # =====================================================
    def _score_level(self, text, keywords):
        score = 0
        for k in keywords:
            if k in text:
                score += 1
        return score

    # =====================================================
    # LDL DISTRIBUTION (CORE)
    # =====================================================
    def predict_distribution(self, question: str):
        text = self._clean(question)

        raw_scores = []

        for level in self.levels:
            score = self._score_level(text, self.patterns[level])
            raw_scores.append(score + 0.1)  # smoothing

        raw_scores = np.array(raw_scores, dtype=np.float32)

        # softmax normalization → probability distribution
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / np.sum(exp_scores)

        return {
            lvl: float(p)
            for lvl, p in zip(self.levels, probs)
        }

    # =====================================================
    # PREDICTION (TOP LEVEL)
    # =====================================================
    def predict(self, question: str):
        dist = self.predict_distribution(question)

        best_label = max(dist, key=dist.get)

        return best_label, dist

    # =====================================================
    # UNCERTAINTY (IMPORTANT FOR THESIS)
    # =====================================================
    def uncertainty(self, dist: dict):

        probs = np.array(list(dist.values()))

        # entropy-based uncertainty
        entropy = -np.sum(probs * np.log(probs + 1e-8))

        # normalize (0–1)
        return float(entropy / np.log(len(probs)))