"""What edgar.py must do. Every test here is `live` — it hits SEC EDGAR.

    pytest tests/test_edgar.py -v
    pytest -m "not live"        # skips all of these

Set SEC_USER_AGENT in .env first (name + email), or SEC returns 403.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sec10k.edgar import FilingRef, cik_for_ticker, download_filing, latest_10k

pytestmark = pytest.mark.live  # mark every test in this module


def test_cik_for_ticker():
    assert cik_for_ticker("AAPL") == 320193
    assert cik_for_ticker("aapl") == 320193  # case-insensitive


def test_cik_for_unknown_ticker_raises():
    with pytest.raises(LookupError):
        cik_for_ticker("NOTATICKER123")


def test_latest_10k_ref():
    ref = latest_10k(320193, ticker="AAPL")
    assert isinstance(ref, FilingRef)
    assert ref.form == "10-K"
    assert "Apple" in ref.company
    assert ref.accession.count("-") == 2
    assert ref.primary_doc.endswith((".htm", ".html"))
    assert ref.doc_url.startswith("https://www.sec.gov/Archives/edgar/data/320193/")
    assert ref.doc_url.endswith(ref.primary_doc)


def test_download_filing(tmp_path: Path):
    ref = latest_10k(320193, ticker="AAPL")
    path = download_filing(ref, dest_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 100_000  # a real 10-K is megabytes
    head = path.read_bytes()[:2000].lower()
    assert b"10-k" in head or b"<html" in head

    # second call must not re-download; same path, still there
    again = download_filing(ref, dest_dir=tmp_path)
    assert again == path
