"""Database access. For now that's exactly one thing: recording LLM calls.

Every LLM request in this project results in one row in `llm_calls`. That table
is the raw material for the cost/latency columns of the evaluation table later,
so the write has to happen on every call — success or failure.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from sec10k.config import get_settings


class LlmCall(BaseModel):
    """One row's worth of data about a single LLM request.

    Built by the LLM client (llm.py) after a response comes back, then handed to
    log_llm_call(). Kept as a model (not loose kwargs) so the same shape is
    validated in one place and reused by tests.
    """

    provider: str
    model: str
    tag: str | None = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0
    latency_ms: int
    temperature: float
    response_id: str | None = None


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a connection to settings.database_url.

    Use as:  with get_connection() as conn: ...
    On a clean exit the transaction commits; on an exception it rolls back.
    (psycopg's own context manager already does this — you just need to open the
    connection with the URL from config and hand it out.)
    """
    with psycopg.connect(get_settings().database_url) as conn:
        # Teach this connection the pgvector types, so Python lists can be passed
        # as query params and `vector` columns come back as lists.
        register_vector(conn)
        yield conn


def log_llm_call(call: LlmCall) -> int:
    """Insert `call` as a row in llm_calls and return the new row's id.

    - Open a connection with get_connection().
    - One INSERT with %s placeholders and a params tuple (no f-strings).
    - Use `RETURNING id` and read it back with fetchone().
    - Return the id as an int.
    """
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO llm_calls (
                provider, model, tag,
                prompt_tokens, completion_tokens, total_tokens,
                cost_usd, latency_ms, temperature, response_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                call.provider,
                call.model,
                call.tag,
                call.prompt_tokens,
                call.completion_tokens,
                call.total_tokens,
                call.cost_usd,
                call.latency_ms,
                call.temperature,
                call.response_id,
            ),
        ).fetchone()
        return row[0]


def record_trace(
    *,
    question: str,
    company: str | None,
    fiscal_year: int | None,
    hybrid: bool,
    reranked: bool,
    retrieved: list[dict],
    answer: str,
    refused: bool,
    llm_call_id: int | None,
    latency_ms: int,
) -> int:
    """Insert one `traces` row (retrieval + generation for a single answer() call)
    and return its id. `retrieved` is a list of small dicts, stored as jsonb."""
    from psycopg.types.json import Json

    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO traces (
                question, company, fiscal_year, hybrid, reranked,
                retrieved, answer, refused, llm_call_id, latency_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                question, company, fiscal_year, hybrid, reranked,
                Json(retrieved), answer, refused, llm_call_id, latency_ms,
            ),
        ).fetchone()
        return row[0]
