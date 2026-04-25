import time
from llama_cpp import Llama


# =====================================================
# MODEL LOADER
# =====================================================
def load_rag_model(path: str):
    return Llama(
        model_path=path,
        n_ctx=4096,
        n_threads=8,
        use_mmap=True,
        use_mlock=False,
        verbose=False
    )


# =====================================================
# CONTEXT SAFE TRUNCATION (IMPORTANT FIX)
# =====================================================
def _truncate(text: str, max_chars: int = 2500):
    if not text:
        return ""
    return text[:max_chars]


# =====================================================
# GENERATION FUNCTION (FIXED)
# =====================================================
def generate(llm, prompt, context=None, temperature=0.2, max_tokens=256):

    context = _truncate(context, 2000)

    # -------------------------
    # SAFE PROMPT BUILDING
    # -------------------------
    if context:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict academic assistant. "
                    "Answer ONLY using provided context. "
                    "If the answer is not in context, say 'Not found in documents'."
                )
            },
            {
                "role": "user",
                "content": f"""
CONTEXT:
{context}

QUESTION:
{prompt}

ANSWER:
"""
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a strict academic assistant."
            },
            {
                "role": "user",
                "content": f"""
QUESTION:
{prompt}

ANSWER:
Not found in documents.
"""
            }
        ]

    # -------------------------
    # GENERATION
    # -------------------------
    start = time.time()

    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return {
        "response": response["choices"][0]["message"]["content"],
        "latency": time.time() - start
    }