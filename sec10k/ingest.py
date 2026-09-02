"""Fetch a company's latest 10-K, chunk it, embed every chunk, store the rows.

The bridge from raw EDGAR filings to something `search()` can query. Composes
edgar -> chunker -> embeddings -> chunks table.
"""
from __future__ import annotations

from pgvector import Vector
from pydantic import BaseModel

from sec10k.chunker import chunk_structured, html_to_blocks
from sec10k.db import get_connection
from sec10k.edgar import cik_for_ticker, download_filing, latest_10k
from sec10k.embeddings import embed_texts

_INSERT = """
    INSERT INTO chunks
        (source, ord, text, embedding, company, cik, accession, fiscal_year, section, kind)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class IngestResult(BaseModel):
    company: str
    cik: int
    accession: str
    fiscal_year: int
    n_chunks: int
    skipped: bool = False  # already present and replace=False


def ingest_ticker(
    ticker: str,
    *,
    before: str | None = None,
    replace: bool = True,
    cik: int | None = None,
) -> IngestResult:
    """Ingest the most recent 10-K for `ticker` into the chunks table.

    `before` (ISO date) picks an older filing; `replace` re-ingests one already
    stored (otherwise an existing accession is left alone and skipped=True).
    `cik` overrides the ticker lookup — needed when SEC's ticker file points at a
    newly-formed holding company that hasn't filed yet.
    """
    ref = latest_10k(cik or cik_for_ticker(ticker), ticker=ticker, before=before)
    fiscal_year = int(ref.report_date[:4])

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT count(*) FROM chunks WHERE accession = %s", (ref.accession,)
        ).fetchone()[0]
        if existing and not replace:
            return IngestResult(
                company=ref.company, cik=ref.cik, accession=ref.accession,
                fiscal_year=fiscal_year, n_chunks=existing, skipped=True,
            )
        if existing:
            conn.execute("DELETE FROM chunks WHERE accession = %s", (ref.accession,))

    html = download_filing(ref).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_structured(html_to_blocks(html))

    # Natural-language header so terse tables still embed/rerank against queries
    # like "NVIDIA total revenue fiscal year 2026".
    def enrich(c) -> str:
        where = c.section or "general section"
        return f"{ref.company} — fiscal year {fiscal_year} — {where} ({c.kind}).\n{c.text}"

    texts = [enrich(c) for c in chunks]
    vectors = embed_texts(texts)

    params = [
        (
            ref.accession, i, text, Vector(v),
            ref.company, ref.cik, ref.accession, fiscal_year,
            c.section or None, c.kind,
        )
        for i, (c, text, v) in enumerate(zip(chunks, texts, vectors))
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(_INSERT, params)

    return IngestResult(
        company=ref.company, cik=ref.cik, accession=ref.accession,
        fiscal_year=fiscal_year, n_chunks=len(chunks),
    )
