"""End-to-end ingestion: EDGAR -> chunker -> embeddings -> chunks table -> search.
Live (hits SEC, downloads a filing, embeds ~150 chunks).

    pytest tests/test_ingest.py -v
"""
from __future__ import annotations

import psycopg
import pytest

from sec10k.config import get_settings
from sec10k.ingest import ingest_ticker
from sec10k.search import search

pytestmark = pytest.mark.live


@pytest.fixture
def apple_ingested():
    res = ingest_ticker("AAPL", replace=True)
    yield res
    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute("DELETE FROM chunks WHERE accession = %s", (res.accession,))


def test_ingest_stores_all_chunks_with_metadata(apple_ingested):
    res = apple_ingested
    assert res.n_chunks > 50
    assert "Apple" in res.company
    assert res.fiscal_year >= 2024

    with psycopg.connect(get_settings().database_url) as conn:
        conn.row_factory = psycopg.rows.dict_row
        row = conn.execute(
            "SELECT count(*) n, count(DISTINCT section) sections, "
            "       count(*) FILTER (WHERE kind = 'table') tables "
            "FROM chunks WHERE accession = %s",
            (res.accession,),
        ).fetchone()

    assert row["n"] == res.n_chunks
    assert row["sections"] > 3          # multiple Item N sections were tagged
    assert row["tables"] > 5            # tables were kept as their own chunks


def test_search_returns_filing_metadata_after_ingest(apple_ingested):
    res = apple_ingested
    hits = search("net sales in Greater China", k=5)
    assert any(h.accession == res.accession for h in hits)
    top = hits[0]
    assert top.company and top.fiscal_year == res.fiscal_year
    assert top.section is not None


def test_reingest_replaces_not_duplicates(apple_ingested):
    res = apple_ingested
    again = ingest_ticker("AAPL", replace=True)
    with psycopg.connect(get_settings().database_url) as conn:
        n = conn.execute(
            "SELECT count(*) FROM chunks WHERE accession = %s", (again.accession,)
        ).fetchone()[0]
    assert n == again.n_chunks  # not 2x
