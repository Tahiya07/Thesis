# src/privacy/privacy_filter.py

class PrivacyFilter:
    def __init__(self):
        self.block_keywords = [
            "student id", "id number", "registration",
            "password", "credential",
            "marks", "grade", "score",
            "exam", "exam sheet", "answer script",
            "confidential"
        ]

    def risk_score(self, text: str) -> float:
        text = text.lower()
        score = 0

        for k in self.block_keywords:
            if k in text:
                score += 1

        return min(score / len(self.block_keywords), 1.0)
