#src/privacy/privacy.py

import hashlib

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def is_private(question):
    private_keywords = ["student id", "exam", "personal", "marks", "name"]
    return any(k in question.lower() for k in private_keywords)