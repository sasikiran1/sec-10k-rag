"""The demo API's non-LLM endpoints. `/ask` is exercised via sec10k.answer's tests.

    pytest tests/test_app.py -v
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from sec10k.app import app

client = TestClient(app)


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "sec-10k-rag" in r.text


def test_filings_lists_corpus(corpus):
    r = client.get("/filings")
    assert r.status_code == 200
    rows = r.json()
    assert rows and all({"company", "fiscal_year"} <= row.keys() for row in rows)
