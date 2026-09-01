"""Naive vector search over the `chunks` table.

Embed text on the way in, embed the query on the way out, let pgvector find the
nearest stored vectors by cosine distance. This is the baseline retrieval that
every later improvement (hybrid, rerank, decomposition) has to beat.
"""
from __future__ import annotations

from pgvector import Vector
from psycopg.rows import dict_row
from pydantic import BaseModel

from sec10k.db import get_connection
from sec10k.embeddings import embed_text, embed_texts


class Hit(BaseModel):
    """One search result. Filing-metadata fields are None for ad-hoc test data."""

    id: int
    source: str
    ord: int
    text: str
    score: float  # cosine similarity in [-1, 1]; higher = more similar
    section: str | None = None
    kind: str | None = None
    company: str | None = None
    accession: str | None = None
    fiscal_year: int | None = None


def add_chunks(rows: list[tuple[str, int, str]]) -> int:
    """Insert chunks. Each row is (source, ord, text); the embedding is computed here.
    Returns the number of rows inserted.

    Implementation:
      1. texts   = [text for (_src, _ord, text) in rows]
      2. vectors = embed_texts(texts)
      3. params  = [(src, ord_, text, vec)
                    for (src, ord_, text), vec in zip(rows, vectors)]
      4. with get_connection() as conn:
             with conn.cursor() as cur:
                 cur.executemany(
                     "INSERT INTO chunks (source, ord, text, embedding) "
                     "VALUES (%s, %s, %s, %s)",
                     params,
                 )
      5. return len(rows)
    """
    texts = [text for (_src, _ord, text) in rows]
    vectors = embed_texts(texts)
    params = [
        (src, ord_, text, Vector(vec))
        for (src, ord_, text), vec in zip(rows, vectors)
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (source, ord, text, embedding) VALUES (%s, %s, %s, %s)",
                params,
            )
    return len(rows)


def search(
    query: str,
    k: int = 5,
    *,
    company: str | None = None,
    fiscal_year: int | None = None,
) -> list[Hit]:
    """Return the k chunks whose embeddings are closest to `query` by cosine.

    pgvector's `<=>` is cosine DISTANCE: 0 = identical direction, 1 = orthogonal,
    2 = opposite. Cosine SIMILARITY = 1 - distance. So we ORDER BY the distance
    ascending (nearest first) and report `1 - distance` as the score.

    `company` / `fiscal_year` restrict the search to one filing's chunks — the eval
    scopes questions this way so an Apple question can't retrieve Microsoft text.

    Rows come back as dicts (psycopg dict_row) so Hit(**row) just works.
    """
    qv = Vector(embed_text(query))
    params: dict = {"qv": qv, "k": k}
    filters = []
    if company is not None:
        filters.append("company = %(company)s")
        params["company"] = company
    if fiscal_year is not None:
        filters.append("fiscal_year = %(fiscal_year)s")
        params["fiscal_year"] = fiscal_year
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    sql = f"""
        SELECT id, source, ord, text, section, kind, company, accession, fiscal_year,
               1 - (embedding <=> %(qv)s) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> %(qv)s
        LIMIT %(k)s
    """
    with get_connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(sql, params).fetchall()
    return [Hit(**row) for row in rows]
