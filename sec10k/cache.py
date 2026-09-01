"""Content-addressed cache for LLM responses.

Temperature is 0 everywhere, so a request always yields the same response. Caching
by a hash of the exact request makes re-running the eval nearly free when nothing
changed; any change to the prompt or retrieved context changes the hash and misses.

On by default. Set LLM_CACHE=0 to bypass (e.g. to force fresh calls).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

_DB = Path(__file__).resolve().parent.parent / ".cache" / "llm.sqlite"


def enabled() -> bool:
    return os.environ.get("LLM_CACHE", "1") != "0"


def _conn() -> sqlite3.Connection:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS responses ("
        " key TEXT PRIMARY KEY,"
        " value TEXT NOT NULL,"
        " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def key_for(payload: dict[str, Any]) -> str:
    """Stable sha256 of a request payload (order-independent)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def get(key: str) -> dict | None:
    if not enabled():
        return None
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM responses WHERE key = ?", (key,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def put(key: str, value: dict) -> None:
    if not enabled():
        return
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
