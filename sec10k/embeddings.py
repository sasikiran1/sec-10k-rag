"""Turn text into vectors with a local sentence-transformers model.

One model, loaded once. Vectors are L2-normalized (length 1), so a dot product
between two of them equals their cosine similarity, and pgvector's cosine
distance behaves predictably.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # this model's output size; the pgvector column must match


@lru_cache
def _model() -> SentenceTransformer:
    """Load (and cache for the process) the embedding model.

    First call downloads weights to the local HF cache; later calls are instant.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts at once.

    Returns one 384-float list per input, in the same order. Use the batch form
    whenever you have more than one string — the model vectorizes a batch far
    faster than N single calls.

    Implementation:
        arr = _model().encode(texts, normalize_embeddings=True)
        return [row.tolist() for row in arr]   # plain python floats, no numpy leaking out
    """
    arr = _model().encode(texts, normalize_embeddings=True)
    return [row.tolist() for row in arr]


def embed_text(text: str) -> list[float]:
    """Embed a single string -> one 384-float list. Thin wrapper over embed_texts."""
    return embed_texts([text])[0]
