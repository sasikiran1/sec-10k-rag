"""What embeddings.py must do. No network calls to our LLM API; the first run
downloads the model to the local HF cache, later runs are offline.

    pytest tests/test_embeddings.py -v
"""
from __future__ import annotations

import math

from sec10k.embeddings import EMBEDDING_DIM, embed_text, embed_texts


def test_embed_text_shape():
    v = embed_text("annual report risk factors")
    assert isinstance(v, list)
    assert len(v) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in v)


def test_vectors_are_unit_length():
    v = embed_text("gross margin improved year over year")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-3


def test_deterministic():
    # The eval harness re-runs; embeddings must not drift between runs.
    a = embed_text("net cash provided by operating activities")
    b = embed_text("net cash provided by operating activities")
    assert a == b


def test_batch_matches_single():
    texts = ["first sentence about revenue", "second sentence about litigation"]
    batch = embed_texts(texts)
    assert len(batch) == 2
    assert all(len(row) == EMBEDDING_DIM for row in batch)
    # Same text embedded in a batch vs alone: equal only up to float noise, because
    # batch shape changes the summation order inside the model. ~1e-8 in practice.
    single = embed_text(texts[0])
    assert all(abs(a - b) < 1e-5 for a, b in zip(batch[0], single))


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # unit vectors -> dot == cosine


def test_semantic_ordering():
    base = embed_text("The company reported a net profit for fiscal 2023.")
    near = embed_text("The firm posted positive earnings in FY2023.")
    far = embed_text("Item 1A describes the risk factors facing the business.")
    assert _cos(base, near) > _cos(base, far)
