#src/privacy/privacy.py

import hashlib


PRIVATE_KEYWORDS = [
    "student id", "id number", "registration",
    "password", "credential",
    "marks", "grade", "score",
    "exam", "confidential",
    "personal", "name"
]

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def is_private(question):
    return any(k in question.lower() for k in PRIVATE_KEYWORDS)
