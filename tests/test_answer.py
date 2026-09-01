"""answer.py: format_context is a unit test; answer() is live (retrieval + LLM).

    pytest tests/test_answer.py -v
"""
from __future__ import annotations

import pytest

from sec10k.answer import answer, format_context
from sec10k.goldens import REFUSAL
from sec10k.search import Hit


def _hit(text: str, n: int) -> Hit:
    return Hit(
        id=n, source="x", ord=n, text=text, score=0.5,
        section="Item 8", company="Apple Inc.", fiscal_year=2025,
    )


def test_format_context_numbers_entries_and_keeps_text():
    ctx = format_context([_hit("alpha figure", 1), _hit("beta figure", 2)])
    assert "[1]" in ctx and "[2]" in ctx
    assert "alpha figure" in ctx and "beta figure" in ctx
    assert "Apple Inc." in ctx and "FY2025" in ctx


def test_format_context_respects_char_budget():
    big = [_hit("x" * 5000, n) for n in range(6)]
    ctx = format_context(big, char_budget=12000)
    assert len(ctx) <= 12100          # a couple of whole chunks, not all six
    assert ctx.count("FY2025") <= 3


def test_format_context_always_includes_top_hit_even_if_huge():
    ctx = format_context([_hit("y" * 40000, 1)], char_budget=1000)
    assert "truncated" in ctx and len(ctx) < 1200


@pytest.mark.live
def test_answer_finds_a_known_figure():
    res = answer("What were Apple's total net sales for fiscal year 2023?")
    assert res.hits, "retrieval returned nothing"
    assert "383" in res.answer  # 383,285 / 383.3 billion, in some form
    assert res.chat.record.total_tokens > 0


@pytest.mark.live
def test_answer_refuses_when_context_lacks_it():
    res = answer("What is the home address of Apple's CFO?")
    assert REFUSAL in res.answer
