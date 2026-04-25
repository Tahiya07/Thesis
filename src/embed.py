import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


_st_model = None
_fallback_vectorizer = HashingVectorizer(
    n_features=384,
    alternate_sign=False,
    norm=None
)


def _load_sentence_transformer():
    global _st_model
    if _st_model is not None:
        return _st_model

    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        _st_model = None

    return _st_model


def _fallback_embed(texts):
    X = _fallback_vectorizer.transform(texts)
    X = normalize(X, norm="l2", axis=1)
    return X.toarray().astype(np.float32)


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    model = _load_sentence_transformer()
    if model is not None:
        return model.encode(texts, normalize_embeddings=True)

    return _fallback_embed(texts)
