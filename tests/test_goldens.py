"""The golden set must load, be internally consistent, and (live) actually point
at facts present in the ingested corpus.

    pytest tests/test_goldens.py -v
"""
from __future__ import annotations

import psycopg
import pytest

from sec10k.config import get_settings
from sec10k.goldens import REFUSAL, Golden, load_goldens


def test_loads_and_validates():
    gs = load_goldens()
    assert len(gs) >= 18
    assert all(isinstance(g, Golden) for g in gs)


def test_ids_are_unique():
    gs = load_goldens()
    ids = [g.id for g in gs]
    assert len(ids) == len(set(ids))


def test_refusal_items_are_shaped_right():
    for g in load_goldens():
        if g.kind == "refusal":
            assert g.answer == REFUSAL
            assert g.must_contain == []
        else:
            assert g.answer != REFUSAL
            assert g.must_contain, f"{g.id} needs must_contain for retrieval scoring"


def test_has_every_kind():
    kinds = {g.kind for g in load_goldens()}
    assert kinds == {"single", "multi_hop", "comparison", "refusal"}


@pytest.mark.live
def test_answers_are_reachable_in_the_corpus():
    """Each non-refusal golden must have at least one chunk in the ingested corpus
    that contains ALL its must_contain strings. Guards against typos in the set.
    Requires: python scripts/build_corpus.py
    """
    missing = []
    with psycopg.connect(get_settings().database_url) as conn:
        for g in load_goldens():
            if g.kind == "refusal":
                continue
            sql = "SELECT count(*) FROM chunks WHERE company = %s AND fiscal_year = %s"
            params: list = [g.company, g.fiscal_year]
            for s in g.must_contain:
                sql += " AND text LIKE %s"
                params.append(f"%{s}%")
            n = conn.execute(sql, params).fetchone()[0]
            if n == 0:
                missing.append(g.id)
    assert not missing, f"no chunk contains all must_contain strings for: {missing}"
