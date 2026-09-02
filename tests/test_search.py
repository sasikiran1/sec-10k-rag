"""What search.py must do. Needs the DB up. Loads the embedding model (slow-ish
first time). No calls to our LLM API.

    pytest tests/test_search.py -v

These tests seed a handful of chunks under a private `source` and scope their
assertions to those rows, so they hold whether or not a real corpus is loaded.
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


def _mine(hits: list[Hit]) -> list[Hit]:
    return [h for h in hits if h.source == SOURCE]


def test_add_chunks_returns_count():
    assert add_chunks([(SOURCE, 0, "hello world")]) == 1


def test_search_returns_k_hits_of_the_right_type():
    _seed()
    hits = search("anything at all", k=3)
    assert len(hits) == 3
    assert all(isinstance(h, Hit) for h in hits)


def test_most_relevant_seeded_chunk_ranks_above_the_others():
    _seed()
    mine = _mine(search("how much revenue did the company report", k=5000))
    assert mine, "seeded rows not returned at all"
    assert "net revenue" in mine[0].text


def test_scores_are_sorted_descending():
    _seed()
    hits = search("dividend paid to shareholders", k=5000)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    mine = _mine(hits)
    assert mine and "dividend" in mine[0].text


def test_company_filter_restricts_results(corpus):
    hits = search("total revenue", k=10, company="MICROSOFT CORP")
    assert hits, "expected Microsoft chunks"
    assert all(h.company == "MICROSOFT CORP" for h in hits)


def test_fiscal_year_filter_restricts_results(corpus):
    hits = search("net sales", k=10, company="Apple Inc.", fiscal_year=2023)
    assert hits
    assert all(h.fiscal_year == 2023 and h.company == "Apple Inc." for h in hits)
