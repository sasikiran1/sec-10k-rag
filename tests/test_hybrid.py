"""keyword_search + hybrid_search. Needs the DB and the ingested corpus.

    pytest tests/test_hybrid.py -v
"""
from __future__ import annotations

from sec10k.search import Hit, hybrid_search, keyword_search, search


def test_keyword_search_finds_exact_token(corpus):
    # "H20" is a distinctive token that lives in exactly one filing's MD&A.
    hits = keyword_search("H20 excess inventory charge", k=5, company="NVIDIA CORP")
    assert hits
    assert any("H20" in h.text for h in hits)


def test_keyword_search_respects_filters(corpus):
    hits = keyword_search("total revenue", k=10, company="MICROSOFT CORP", fiscal_year=2026)
    assert hits
    assert all(h.company == "MICROSOFT CORP" and h.fiscal_year == 2026 for h in hits)


def test_hybrid_returns_k_hits(corpus):
    hits = hybrid_search("Greater China net sales", k=6, company="Apple Inc.", fiscal_year=2023)
    assert len(hits) == 6
    assert all(isinstance(h, Hit) for h in hits)


def test_hybrid_scores_are_descending(corpus):
    hits = hybrid_search("research and development expense", k=8, company="Apple Inc.", fiscal_year=2025)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_keyword_search_matches_on_any_term_not_all(corpus):
    # An OR query: a chunk that has "dividend" but not the other words still matches.
    hits = keyword_search("quarterly dividend declared per share", k=10, company="Apple Inc.", fiscal_year=2023)
    assert hits
    assert any("dividend" in h.text.lower() for h in hits)


def test_hybrid_merges_both_sources(corpus):
    q = "Greater China net sales 72,559"
    vec = {h.id for h in search(q, k=20, company="Apple Inc.", fiscal_year=2023)}
    kw = {h.id for h in keyword_search(q, k=20, company="Apple Inc.", fiscal_year=2023)}
    hyb = {h.id for h in hybrid_search(q, k=10, company="Apple Inc.", fiscal_year=2023)}
    # every hybrid hit came from one of the two pools
    assert hyb and hyb <= (vec | kw)
    # and hybrid draws from both, not just one
    assert hyb & vec and hyb & kw
