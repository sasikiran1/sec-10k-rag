"""What search.py must do. Needs the DB up. Loads the embedding model (slow-ish
first time). No calls to our LLM API.

    pytest tests/test_search.py -v
"""
from __future__ import annotations

import pytest

from sec10k.db import get_connection
from sec10k.search import Hit, add_chunks, search

SOURCE = "pytest-search"


@pytest.fixture(autouse=True)
def _clean_after():
    yield
    with get_connection() as conn:
        conn.execute("DELETE FROM chunks WHERE source = %s", (SOURCE,))


def _seed() -> None:
    add_chunks([
        (SOURCE, 0, "Total net revenue increased 8% to $394.3 billion for fiscal 2023."),
        (SOURCE, 1, "The board declared a quarterly cash dividend of $0.24 per share."),
        (SOURCE, 2, "Our principal executive offices are located in Cupertino, California."),
        (SOURCE, 3, "Item 1A. Risk Factors. These risks could materially affect our business."),
    ])


def test_add_chunks_returns_count():
    assert add_chunks([(SOURCE, 0, "hello world")]) == 1


def test_search_returns_k_hits_of_the_right_type():
    _seed()
    hits = search("anything at all", k=3)
    assert len(hits) == 3
    assert all(isinstance(h, Hit) for h in hits)


def test_most_relevant_chunk_ranks_first():
    _seed()
    hits = search("how much revenue did the company report", k=4)
    assert "net revenue" in hits[0].text


def test_scores_are_sorted_descending():
    _seed()
    hits = search("dividend paid to shareholders", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert "dividend" in hits[0].text
