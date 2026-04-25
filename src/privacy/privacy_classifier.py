import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class PrivacyClassifier:

    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        self.trained = False

    # ---------------------------------------
    # TRAINING DATA (you can expand later)
    # ---------------------------------------
    def train(self, texts, labels):
        """
        labels:
        1 = private/sensitive
        0 = safe
        """

        self.model.fit(texts, labels)
        self.trained = True

    # ---------------------------------------
    # PREDICT PRIVACY PROBABILITY
    # ---------------------------------------
    def predict_risk(self, text: str) -> float:

        if not self.trained:
            return 0.0  # safe fallback

        prob = self.model.predict_proba([text])[0][1]
        return float(prob)

    # ---------------------------------------
    # HARD CLASSIFICATION
    # ---------------------------------------
    def is_private(self, text: str, threshold=0.5) -> bool:
        return self.predict_risk(text) >= threshold