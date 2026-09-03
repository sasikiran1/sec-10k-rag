"""Minimal web demo: ask a question, see the answer and the exact chunks it used.

    uvicorn sec10k.app:app --reload    # then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import APIError
from pydantic import BaseModel

from sec10k.answer import answer
from sec10k.db import get_connection

app = FastAPI(title="sec-10k-rag")
_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


class Ask(BaseModel):
    question: str
    company: str | None = None
    fiscal_year: int | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX)


@app.get("/filings")
def filings() -> list[dict]:
    """Distinct (company, fiscal_year) in the corpus, for the scope dropdown."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT company, fiscal_year FROM chunks "
            "WHERE company IS NOT NULL ORDER BY company, fiscal_year"
        ).fetchall()
    return [{"company": c, "fiscal_year": y} for c, y in rows]


@app.post("/ask")
def ask(body: Ask) -> dict:
    try:
        r = answer(
            body.question, company=body.company, fiscal_year=body.fiscal_year,
            max_retries=2,  # interactive: fail fast rather than sit through backoff
            cite=True,      # ask for [n] references; the eval runs without this
        )
    except APIError as e:
        msg = getattr(e, "message", str(e))
        return {"error": f"LLM call failed ({type(e).__name__}): {msg}"}
    # gpt-oss sometimes emits full-width brackets for citations; normalize to [n].
    text = r.answer.translate(str.maketrans({"【": "[", "】": "]"}))
    return {
        "answer": text.strip(),
        "tokens": r.chat.record.total_tokens,
        "latency_ms": r.chat.record.latency_ms,
        "cached": r.chat.cached,
        "hits": [
            {
                "score": round(h.score, 3),
                "section": h.section,
                "company": h.company,
                "fiscal_year": h.fiscal_year,
                "kind": h.kind,
                "text": h.text,
            }
            for h in r.hits
        ],
    }
