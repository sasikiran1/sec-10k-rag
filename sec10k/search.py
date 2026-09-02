"""Naive vector search over the `chunks` table.

Embed text on the way in, embed the query on the way out, let pgvector find the
nearest stored vectors by cosine distance. This is the baseline retrieval that
every later improvement (hybrid, rerank, decomposition) has to beat.
"""
from __future__ import annotations

from collections import defaultdict

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
    """Insert (source, ord, text) rows, embedding the text here. Returns the count.
    Filing metadata columns are left NULL — see sec10k.ingest for the real path."""
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
    """Postgres full-text search over chunks.text (BM25-ish via ts_rank_cd)."""
    params: dict = {"q": query, "k": k}
    # websearch_to_tsquery ANDs every term; rewrite '&' -> '|' so a chunk matching
    # ANY query term is a candidate, ranked by ts_rank_cd (term coverage + rarity).
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
    """Fuse the vector and keyword rankings with Reciprocal Rank Fusion:
    each chunk scores sum(1 / (rrf_k + rank)) across the two lists.
    (Measured worse than vector alone — kept for the ablation. See evals/ablation.md.)"""
    v = search(query, k=pool, company=company, fiscal_year=fiscal_year)
    t = keyword_search(query, k=pool, company=company, fiscal_year=fiscal_year)

    by_id: dict[int, Hit] = {h.id: h for h in v + t}
    score: dict[int, float] = defaultdict(float)
    for lst in (v, t):
        for rank, h in enumerate(lst, start=1):
            score[h.id] += 1.0 / (rrf_k + rank)

    top_ids = sorted(score, key=score.get, reverse=True)[:k]
    return [by_id[i].model_copy(update={"score": score[i]}) for i in top_ids]
