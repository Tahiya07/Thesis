# src/chunker.py
import re

def chunk_text(text, max_words=180, overlap=40):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = []

    word_count = 0

    for s in sentences:
        wc = len(s.split())

        if word_count + wc > max_words:
            chunks.append(" ".join(current))

            # overlap
            current = current[-overlap:] if overlap < len(current) else current
            word_count = sum(len(x.split()) for x in current)

        current.append(s)
        word_count += wc

    if current:
        chunks.append(" ".join(current))

    return chunks