"""Cross-encoder reranking.

The bi-encoder (sentence-transformers) scores query and chunk *independently* then
compares vectors — it can't model how specific query words relate to specific
chunk words. A cross-encoder reads (query, chunk) *together* with full attention
and outputs one relevance score. Far more accurate, but one forward pass per
candidate, so it re-ranks the bi-encoder's top-N rather than searching.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from sec10k.search import Hit

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def _model() -> CrossEncoder:
    """Load (and process-cache) the cross-encoder. First call downloads ~80 MB."""
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, hits: list[Hit], *, top_k: int) -> list[Hit]:
    """Re-score `hits` against `query` with the cross-encoder; return the top_k.

    Implementation:
      1. if not hits: return []
      2. pairs  = [(query, h.text) for h in hits]
      3. scores = _model().predict(pairs)                 # np array, higher = better
      4. ranked = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
      5. return [h.model_copy(update={"score": float(s)}) for h, s in ranked[:top_k]]
    """
    if not hits:
        return []
    pairs = [(query, h.text) for h in hits]
    scores = _model().predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
    return [h.model_copy(update={"score": float(s)}) for h, s in ranked[:top_k]]
