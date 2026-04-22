# src/embed.py

from typing import List, Union

_model = None


def _get_model():
    """
    Lazy-load SentenceTransformer only when first needed.
    Prevents slow startup and unwanted import-time loading.
    """
    global _model

    if _model is None:
        print("🧠 Loading embedding model (lazy)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def embed(texts: Union[str, List[str]]):
    """
    Safe embedding function (lazy-loaded model)

    Args:
        texts: str or list of strings

    Returns:
        numpy array embeddings
    """

    if isinstance(texts, str):
        texts = [texts]

    model = _get_model()

    return model.encode(texts, normalize_embeddings=True)