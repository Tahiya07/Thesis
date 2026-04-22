#src/summarizer.py

def summarize(llm, text: str):
    """
    Lightweight summarization for RAG compression layer.
    Works with llama-cpp Qwen GGUF model.
    """

    prompt = f"""
You are a precise academic summarizer.

Task:
Summarize the following text into a compact academic representation.

Rules:
- Keep meaning intact
- Remove redundancy
- 2-4 sentences max

Text:
{text}

Summary:
"""

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a strict summarization model."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        # SAFE FALLBACK (VERY IMPORTANT FOR RAG PIPELINE)
        return text