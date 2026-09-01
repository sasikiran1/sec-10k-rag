"""Metric math for evaluate.py. Pure functions, no network, no DB.

    pytest tests/test_evaluate.py -v
"""
from __future__ import annotations

from sec10k.evaluate import ItemResult, aggregate, is_relevant, score_retrieval
from sec10k.goldens import Golden
from sec10k.search import Hit


def _hit(text: str, n: int = 1) -> Hit:
    return Hit(id=n, source="s", ord=n, text=text, score=0.5)


def _golden(**kw) -> Golden:
    base = dict(
        id="g", question="q", answer="a", company="Apple Inc.",
        fiscal_year=2023, kind="single", must_contain=["Total net sales", "383,285"],
    )
    base.update(kw)
    return Golden(**base)


def test_is_relevant_needs_all_substrings():
    g = _golden()
    assert is_relevant(_hit("... Total net sales 383,285 ..."), g)
    assert not is_relevant(_hit("... Total net sales 999 ..."), g)
    assert not is_relevant(_hit("nothing useful"), g)


def test_score_retrieval_hit_at_rank_2():
    g = _golden()
    hits = [_hit("irrelevant", 1), _hit("Total net sales 383,285", 2), _hit("x", 3)]
    ranks, recall, rr = score_retrieval(hits, g)
    assert ranks == [2]
    assert recall == 1.0
    assert rr == 0.5


def test_score_retrieval_no_relevant_hit():
    g = _golden()
    ranks, recall, rr = score_retrieval([_hit("a", 1), _hit("b", 2)], g)
    assert ranks == []
    assert recall == 0.0
    assert rr == 0.0


def _item(kind: str, correct: bool, recall, rr, at=10, jt=5) -> ItemResult:
    return ItemResult(
        id="x", kind=kind, question="q", gold="g", predicted="p",
        correct=correct, score=1.0 if correct else 0.0, judge_reasoning="",
        retrieved=[1, 2], relevant_ranks=[1] if rr else [],
        recall_at_k=recall, reciprocal_rank=rr, answer_tokens=at, judge_tokens=jt,
    )


def test_aggregate_rolls_up():
    items = [
        _item("single", True, 1.0, 1.0),
        _item("single", False, 0.0, 0.0),
        _item("comparison", True, 1.0, 0.5),
        _item("refusal", True, None, None),
    ]
    run = aggregate("baseline", k=6, scoped=False, items=items)

    assert run.n == 4
    assert run.accuracy == 0.75                       # 3 of 4
    assert run.accuracy_by_kind["single"] == 0.5
    assert run.accuracy_by_kind["refusal"] == 1.0
    # recall/mrr average only the 3 non-refusal items
    assert abs(run.recall_at_k - (2 / 3)) < 1e-9
    assert abs(run.mrr - (1.0 + 0.0 + 0.5) / 3) < 1e-9
    assert run.total_tokens == 4 * (10 + 5)
    assert run.label == "baseline" and run.k == 6 and run.scoped is False
