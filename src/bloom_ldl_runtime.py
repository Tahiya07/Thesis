import os
import pickle
import numpy as np


class BloomLDL:

    def __init__(self, model_path="models/bloom_ldl.pkl", min_validation_accuracy=0.25):
        self.levels = [
            "Remember",
            "Understand",
            "Apply",
            "Analyze",
            "Evaluate",
            "Create"
        ]
        self.model_path = model_path
        self.min_validation_accuracy = min_validation_accuracy
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.validation_accuracy = None
        self.patterns = {
            "Remember": ["define", "list", "what is", "who is", "when"],
            "Understand": ["explain", "describe", "summarize"],
            "Apply": ["use", "solve", "calculate", "demonstrate"],
            "Analyze": ["compare", "analyze", "differentiate", "why"],
            "Evaluate": ["evaluate", "critique", "assess", "justify"],
            "Create": ["design", "propose", "develop", "formulate"]
        }
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            return

        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)

            self.validation_accuracy = payload.get("validation_accuracy")
            if (
                self.validation_accuracy is not None and
                self.validation_accuracy < self.min_validation_accuracy
            ):
                return

            self.vectorizer = payload["vectorizer"]
            self.model = payload["model"]
            self.levels = payload.get("levels", self.levels)
            self.is_trained = True
        except Exception:
            self.vectorizer = None
            self.model = None
            self.is_trained = False

    def _clean(self, text):
        return (text or "").lower().strip()

    def _heuristic_distribution(self, question: str):
        text = self._clean(question)
        raw_scores = []

        for level in self.levels:
            score = sum(1 for keyword in self.patterns.get(level, []) if keyword in text)
            raw_scores.append(score + 0.1)

        raw_scores = np.array(raw_scores, dtype=np.float32)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / np.sum(exp_scores)

        return {
            lvl: float(p)
            for lvl, p in zip(self.levels, probs)
        }

    def predict_distribution(self, question: str):
        text = (question or "").strip()

        if self.is_trained and self.vectorizer is not None and self.model is not None:
            X = self.vectorizer.transform([text])
            probs = self.model.predict_proba(X)[0]
            return {
                lvl: float(p)
                for lvl, p in zip(self.levels, probs)
            }

        return self._heuristic_distribution(text)

    def predict(self, question: str):
        dist = self.predict_distribution(question)
        best_label = max(dist, key=dist.get)
        return best_label, dist

    def uncertainty(self, dist: dict):
        probs = np.array(list(dist.values()), dtype=np.float32)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(entropy / np.log(len(probs)))

    def confidence(self, dist: dict):
        if not dist:
            return 0.0
        return float(max(dist.values()))

    def reject(self, dist: dict, confidence_threshold=0.40, uncertainty_threshold=0.80):
        confidence = self.confidence(dist)
        uncertainty = self.uncertainty(dist)
        return bool(confidence < confidence_threshold or uncertainty > uncertainty_threshold)
