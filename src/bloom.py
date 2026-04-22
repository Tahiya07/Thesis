import re
from src.llm import generate

BLOOM_CLASSES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def classify_bloom(llm, question: str):
    """
    Temporary LLM-based Bloom classifier.
    Designed to be replaced later with trained LDL/CORAL model.
    """

    prompt = f"""
You are a strict academic classifier.

Classify the question into ONE Bloom's Taxonomy level.

Valid labels:
Remember, Understand, Apply, Analyze, Evaluate, Create

RULES:
- Output ONLY ONE WORD
- No punctuation
- No explanation

Question:
{question}

Answer:
"""

    result = generate(llm, prompt, max_tokens=5, temperature=0.0)

    text = result["response"].strip()

    # =========================
    # STEP 1: STRICT MATCH
    # =========================
    text_lower = text.lower()

    for label in BLOOM_CLASSES:
        if re.search(rf"\b{label.lower()}\b", text_lower):
            return label

    # =========================
    # STEP 2: SOFT MATCH (robust fallback)
    # =========================
    for label in reversed(BLOOM_CLASSES):
        if label.lower() in text_lower:
            return label

    # =========================
    # STEP 3: SAFE DEFAULT
    # =========================
    return "Understand"