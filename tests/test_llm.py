"""Defines what chat() must do. Needs the DB up AND a real GROQ_API_KEY, because
it makes one live call:

    pytest tests/test_llm.py -v          # runs it
    pytest -m "not live"                 # skips anything marked live

The single test is marked `live` so the rest of the suite can run offline.
"""
from __future__ import annotations

import psycopg
import pytest

from sec10k.config import get_settings
from sec10k.llm import ChatResult, chat


def _delete(row_id: int) -> None:
    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute("DELETE FROM llm_calls WHERE id = %s", (row_id,))


@pytest.mark.live
def test_chat_answers_and_logs_one_row():
    result = chat(
        [
            {"role": "system", "content": "Reply with exactly one word."},
            {"role": "user", "content": "Say: pong"},
        ],
        tag="pytest-smoke",
    )

    # shape of the return value
    assert isinstance(result, ChatResult)
    assert result.text.strip() != ""
    assert isinstance(result.db_id, int)

    # the record it built
    r = result.record
    assert r.provider == "groq"
    assert r.prompt_tokens > 0
    assert r.completion_tokens > 0
    assert r.total_tokens == r.prompt_tokens + r.completion_tokens
    assert r.latency_ms > 0
    assert r.temperature == 0.0

    # the row that actually landed in Postgres
    try:
        with psycopg.connect(get_settings().database_url) as conn:
            conn.row_factory = psycopg.rows.dict_row
            row = conn.execute(
                "SELECT * FROM llm_calls WHERE id = %s", (result.db_id,)
            ).fetchone()

        assert row is not None
        assert row["tag"] == "pytest-smoke"
        assert row["response_id"] == r.response_id
        assert row["total_tokens"] == r.total_tokens
        assert row["created_at"] is not None
    finally:
        _delete(result.db_id)
