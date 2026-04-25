# src/privacy/privacy_filter.py

class PrivacyFilter:
    def __init__(self):
        self.block_keywords = [
            "student id", "password", "marks",
            "exam sheet", "answer script",
            "confidential", "grade"
        ]

    def risk_score(self, text: str) -> float:
        text = text.lower()
        score = 0

        for k in self.block_keywords:
            if k in text:
                score += 1

        return min(score / len(self.block_keywords), 1.0)