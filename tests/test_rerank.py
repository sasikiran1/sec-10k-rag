"""Cross-encoder reranking. Downloads ~80 MB on first run; then offline.

    pytest tests/test_rerank.py -v
"""
from __future__ import annotations

from sec10k.rerank import rerank
from sec10k.search import Hit


def _hit(text: str, n: int) -> Hit:
    return Hit(id=n, source="s", ord=n, text=text, score=0.0)


def test_rerank_empty():
    assert rerank("anything", [], top_k=5) == []


def test_rerank_promotes_the_answer_over_a_question_echo():
    q = "What drove the change in Apple's gross margin?"
    hits = [
        _hit("Item 7 includes a discussion of the principal factors affecting our gross margin.", 1),
        _hit("Gross margin rose to 44.1% from 43.3%, driven by a mix shift toward Services.", 2),
        _hit("The board declared a cash dividend of $0.24 per share.", 3),
    ]
    out = rerank(q, hits, top_k=2)
    assert len(out) == 2
    assert "44.1%" in out[0].text            # the real answer, not the echo
    assert out[0].score >= out[1].score      # scores are the CE relevance, sorted


def test_rerank_caps_at_top_k():
    hits = [_hit(f"chunk number {i}", i) for i in range(10)]
    assert len(rerank("chunk", hits, top_k=3)) == 3
