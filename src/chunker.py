import re

def chunk_text(text, max_words=120, overlap=20):

    text = re.sub(r'\s+', ' ', text)

    # stronger structural splitting (IMPORTANT)
    sections = re.split(r'\n\d+\.\s+|\n[A-Z][^.]{5,}\n', text)

    chunks = []

    for sec in sections:

        sentences = re.split(r'(?<=[.!?])\s+', sec)

        current = []
        count = 0

        for s in sentences:

            words = len(s.split())

            if count + words > max_words:

                if current:
                    chunks.append(" ".join(current))

                current = current[-overlap:]
                count = sum(len(x.split()) for x in current)

            current.append(s)
            count += words

        if current:
            chunks.append(" ".join(current))

    return [c.strip() for c in chunks if len(c.strip()) > 60]