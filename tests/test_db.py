"""Defines what db.py must do. Needs the Postgres container running:

    docker compose up -d
    pytest tests/test_db.py -v

These are integration tests — they hit the real local database on purpose. Each
test cleans up the row it created.
"""
from __future__ import annotations

import psycopg
import pytest

from sec10k.config import get_settings
from sec10k.db import LlmCall, get_connection, log_llm_call, record_trace


@pytest.fixture
def sample_call() -> LlmCall:
    return LlmCall(
        provider="groq",
        model="llama-3.3-70b-versatile",
        tag="test",
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        cost_usd=0.0,
        latency_ms=123,
        temperature=0.0,
        response_id="resp_abc123",
    )


def _delete(row_id: int) -> None:
    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute("DELETE FROM llm_calls WHERE id = %s", (row_id,))


def test_get_connection_yields_usable_connection():
    with get_connection() as conn:
        row = conn.execute("SELECT 1").fetchone()
    assert row[0] == 1


def test_log_llm_call_returns_int_id(sample_call):
    row_id = log_llm_call(sample_call)
    try:
        assert isinstance(row_id, int)
    finally:
        _delete(row_id)


def test_log_llm_call_persists_all_fields(sample_call):
    row_id = log_llm_call(sample_call)
    try:
        with psycopg.connect(get_settings().database_url) as conn:
            conn.row_factory = psycopg.rows.dict_row
            row = conn.execute(
                "SELECT * FROM llm_calls WHERE id = %s", (row_id,)
            ).fetchone()

        assert row["provider"] == "groq"
        assert row["model"] == "llama-3.3-70b-versatile"
        assert row["tag"] == "test"
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 7
        assert row["total_tokens"] == 18
        assert float(row["cost_usd"]) == 0.0
        assert row["latency_ms"] == 123
        assert abs(row["temperature"] - 0.0) < 1e-6
        assert row["response_id"] == "resp_abc123"
        assert row["created_at"] is not None
    finally:
        _delete(row_id)


def test_record_trace_roundtrips_retrieved_jsonb():
    tid = record_trace(
        question="q?", company="Apple Inc.", fiscal_year=2025,
        hybrid=False, reranked=True,
        retrieved=[{"chunk_id": 1, "score": 0.9, "section": "Item 8", "kind": "table"}],
        answer="$1 million", refused=False, llm_call_id=None, latency_ms=42,
    )
    try:
        with psycopg.connect(get_settings().database_url) as conn:
            conn.row_factory = psycopg.rows.dict_row
            row = conn.execute("SELECT * FROM traces WHERE id = %s", (tid,)).fetchone()
        assert row["question"] == "q?"
        assert row["reranked"] is True and row["refused"] is False
        assert row["retrieved"][0]["chunk_id"] == 1        # jsonb -> python list/dict
        assert row["latency_ms"] == 42
    finally:
        with psycopg.connect(get_settings().database_url) as conn:
            conn.execute("DELETE FROM traces WHERE id = %s", (tid,))


def test_log_llm_call_allows_null_tag_and_response_id(sample_call):
    sample_call.tag = None
    sample_call.response_id = None
    row_id = log_llm_call(sample_call)
    try:
        with psycopg.connect(get_settings().database_url) as conn:
            conn.row_factory = psycopg.rows.dict_row
            row = conn.execute(
                "SELECT tag, response_id FROM llm_calls WHERE id = %s", (row_id,)
            ).fetchone()
        assert row["tag"] is None
        assert row["response_id"] is None
    finally:
        _delete(row_id)
