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
    ticker: str, *, before: str | None = None, replace: bool = True
) -> IngestResult:
    """Ingest the most recent 10-K for `ticker` into the chunks table.

    `before` (ISO date) picks an older filing; `replace` re-ingests one already
    stored (otherwise an existing accession is left alone and skipped=True).
    """
    ref = latest_10k(cik_for_ticker(ticker), ticker=ticker, before=before)
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
    vectors = embed_texts([c.text for c in chunks])

    params = [
        (
            ref.accession, i, c.text, Vector(v),
            ref.company, ref.cik, ref.accession, fiscal_year,
            c.section or None, c.kind,
        )
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(_INSERT, params)

    return IngestResult(
        company=ref.company, cik=ref.cik, accession=ref.accession,
        fiscal_year=fiscal_year, n_chunks=len(chunks),
    )
