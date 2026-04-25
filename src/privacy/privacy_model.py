# src/privacy/privacy_model.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class PrivacyClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.model = LogisticRegression(max_iter=200)

        self.trained = False

    def train(self, texts, labels):
        """
        labels:
        1 = private
        0 = safe
        """

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.trained = True

    def predict_proba(self, text: str) -> float:

        if not self.trained:
            return 0.0

        X = self.vectorizer.transform([text])
        return float(self.model.predict_proba(X)[0][1])


def build_default_privacy_model():
    """
    Bootstrap a tiny lightweight classifier so the learned privacy
    path is available in the default runtime system.
    """

    texts = [
        "student id 221 registration number and exam marks are confidential",
        "password reset credential for faculty portal",
        "grade sheet with score and exam result",
        "internal answer script and confidential academic record",
        "registration information with student id and marks",
        "course outline for digital logic design",
        "introduction to thermodynamics and heat transfer",
        "lecture notes on probability and statistics",
        "how does the esp32 control sensors in the robot",
        "academic summary of database normalization concepts",
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    clf = PrivacyClassifier()
    clf.train(texts, labels)
    return clf
