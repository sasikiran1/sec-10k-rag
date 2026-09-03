"""Dense, lexical, and hybrid retrieval over SEC filing chunks."""
from __future__ import annotations

from collections import defaultdict

from pgvector import Vector
from psycopg.rows import dict_row
from pydantic import BaseModel

from sec10k.db import get_connection
from sec10k.embeddings import embed_text, embed_texts


class Hit(BaseModel):
    """A ranked retrieval result."""

    id: int
    source: str
    ord: int
    text: str
    score: float
    section: str | None = None
    kind: str | None = None
    company: str | None = None
    accession: str | None = None
    fiscal_year: int | None = None


def add_chunks(rows: list[tuple[str, int, str]]) -> int:
    """Embed and insert `(source, ord, text)` rows."""
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
    """Return the `k` nearest chunks by cosine similarity.

    Optional company and fiscal-year filters restrict retrieval to one filing.
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


_COLS = "id, source, ord, text, section, kind, company, accession, fiscal_year"


def _filter_sql(company: str | None, fiscal_year: int | None, params: dict) -> str:
    clauses = []
    if company is not None:
        clauses.append("company = %(company)s")
        params["company"] = company
    if fiscal_year is not None:
        clauses.append("fiscal_year = %(fiscal_year)s")
        params["fiscal_year"] = fiscal_year
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def keyword_search(
    query: str,
    k: int = 5,
    *,
    company: str | None = None,
    fiscal_year: int | None = None,
) -> list[Hit]:
    """Run PostgreSQL full-text search ranked with `ts_rank_cd`."""
    params: dict = {"q": query, "k": k}
    # Expand the default AND query to OR so any query term may produce a candidate.
    sql = f"""
        SELECT {_COLS}, ts_rank_cd(text_tsv, q.tsq) AS score
        FROM chunks,
             LATERAL (
                 SELECT replace(websearch_to_tsquery('english', %(q)s)::text, '&', '|')::tsquery AS tsq
             ) q
        WHERE text_tsv @@ q.tsq
              {_filter_sql(company, fiscal_year, params)}
        ORDER BY score DESC
        LIMIT %(k)s
    """
    with get_connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(sql, params).fetchall()
    return [Hit(**row) for row in rows]


def hybrid_search(
    query: str,
    k: int = 5,
    *,
    company: str | None = None,
    fiscal_year: int | None = None,
    pool: int = 20,
    rrf_k: int = 60,
) -> list[Hit]:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion.

    This configuration underperformed dense retrieval in the measured ablation and
    is retained for reproducibility. See `evals/ablation.md`.
    """
    v = search(query, k=pool, company=company, fiscal_year=fiscal_year)
    t = keyword_search(query, k=pool, company=company, fiscal_year=fiscal_year)

    by_id: dict[int, Hit] = {h.id: h for h in v + t}
    score: dict[int, float] = defaultdict(float)
    for lst in (v, t):
        for rank, h in enumerate(lst, start=1):
            score[h.id] += 1.0 / (rrf_k + rank)

    top_ids = sorted(score, key=score.get, reverse=True)[:k]
    return [by_id[i].model_copy(update={"score": score[i]}) for i in top_ids]
