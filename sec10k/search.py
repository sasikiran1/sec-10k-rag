"""Naive vector search over the `chunks` table.

Embed text on the way in, embed the query on the way out, let pgvector find the
nearest stored vectors by cosine distance. This is the baseline retrieval that
every later improvement (hybrid, rerank, decomposition) has to beat.
"""
from __future__ import annotations

from pgvector import Vector
from pydantic import BaseModel

from sec10k.db import get_connection
from sec10k.embeddings import embed_text, embed_texts


class Hit(BaseModel):
    """One search result."""

    id: int
    source: str
    ord: int
    text: str
    score: float  # cosine similarity in [-1, 1]; higher = more similar


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


def search(query: str, k: int = 5) -> list[Hit]:
    """Return the k chunks whose embeddings are closest to `query` by cosine.

    pgvector's `<=>` is cosine DISTANCE: 0 = identical direction, 1 = orthogonal,
    2 = opposite. Cosine SIMILARITY = 1 - distance. So we ORDER BY the distance
    ascending (nearest first) and report `1 - distance` as the score.

    Implementation:
      1. qv = embed_text(query)
      2. sql = '''
             SELECT id, source, ord, text, 1 - (embedding <=> %(qv)s) AS score
             FROM chunks
             ORDER BY embedding <=> %(qv)s
             LIMIT %(k)s
         '''
      3. with get_connection() as conn:
             rows = conn.execute(sql, {"qv": qv, "k": k}).fetchall()
      4. return [Hit(id=r[0], source=r[1], ord=r[2], text=r[3], score=r[4]) for r in rows]
    """
    qv = Vector(embed_text(query))
    sql = """
        SELECT id, source, ord, text, 1 - (embedding <=> %(qv)s) AS score
        FROM chunks
        ORDER BY embedding <=> %(qv)s
        LIMIT %(k)s
    """
    with get_connection() as conn:
        rows = conn.execute(sql, {"qv": qv, "k": k}).fetchall()
    return [Hit(id=r[0], source=r[1], ord=r[2], text=r[3], score=r[4]) for r in rows]
