#src/embed.py

from sentence_transformers import SentenceTransformer


# Load once globally (VERY IMPORTANT)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """
    Safe embedding function
    Supports string or list
    """
    if isinstance(texts, str):
        texts = [texts]

    return _model.encode(texts, normalize_embeddings=True)