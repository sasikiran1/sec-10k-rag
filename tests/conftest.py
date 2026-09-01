"""Shared pytest fixtures."""
from __future__ import annotations

import psycopg
import pytest

from sec10k.config import get_settings


@pytest.fixture(scope="session")
def corpus() -> int:
    """Skip the test unless the evaluation corpus has been ingested.

    Run `python scripts/build_corpus.py` to populate it.
    """
    with psycopg.connect(get_settings().database_url) as conn:
        n = conn.execute(
            "SELECT count(*) FROM chunks WHERE company IS NOT NULL"
        ).fetchone()[0]
    if n == 0:
        pytest.skip("eval corpus not loaded — run: python scripts/build_corpus.py")
    return n
