# src/llm.py

import time
from llama_cpp import Llama


# =========================================================
# 1. LOAD QWEN GGUF (ONLY MODEL YOU NEED)
# =========================================================

def load_rag_model(path: str):
    return Llama(
        model_path=path,
        n_ctx=2084,
        n_threads=8,
        use_mmap=True,
        use_mlock=False,
        verbose=False
    )


# =========================================================
# 2. BLOOM CLASSIFICATION (ZERO-SHOT - IMPORTANT FIX)
# =========================================================

def classify_bloom(llm, question: str):

    prompt = f"""
You are a strict academic classifier.

Classify the question into ONE Bloom level:

Remember, Understand, Apply, Analyze, Evaluate, Create

Rules:
- Output ONLY the label
- No explanation

Question:
{question}

Answer:
"""

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a classification system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=10
    )

    text = response["choices"][0]["message"]["content"]

    for label in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
        if label.lower() in text.lower():
            return label

    return "Understand"


# =========================================================
# 3. GENERATION WRAPPER
# =========================================================

def generate(llm, prompt, context=None, temperature=0.2, max_tokens=512):
    if context:
        final_prompt = f"""
    You are a precise academic assistant.

    RULES:
    - OCR text is more reliable than captions
    - Image captions may be incorrect
    - If uncertain, explicitly say "uncertain"
    - Do NOT hallucinate

    Context:
    {context}

    Question:
    {prompt}

    Answer:
    """
    else:
        final_prompt = prompt

    start = time.time()

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a precise academic assistant."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    end = time.time()

    return {
        "response": response["choices"][0]["message"]["content"],
        "latency": end - start,
        "tokens": len(response["choices"][0]["message"]["content"].split())
    }