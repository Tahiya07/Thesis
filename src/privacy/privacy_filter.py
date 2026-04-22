# src/privacy/privacy_filter.py

class PrivacyFilter:
    def __init__(self):
        self.block_keywords = [
            "student id", "password", "marks", "exam sheet", "private"
        ]

    def classify(self, text: str):
        text_lower = text.lower()

        if any(k in text_lower for k in self.block_keywords):
            return "PRIVATE"

        return "PUBLIC"