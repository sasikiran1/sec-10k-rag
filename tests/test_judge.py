"""judge.py — all live (each test is one structured LLM call).

    pytest tests/test_judge.py -v
"""
from __future__ import annotations

import pytest

from sec10k.goldens import REFUSAL
from sec10k.judge import Verdict, judge

pytestmark = pytest.mark.live


def test_same_number_different_format_is_correct():
    v = judge(
        "What were Apple's total net sales for fiscal 2023?",
        "$383,285 million",
        "Apple's net sales were about $383.3 billion.",
    )
    assert isinstance(v, Verdict)
    assert v.correct is True


def test_wrong_number_is_incorrect():
    v = judge(
        "What were Apple's total net sales for fiscal 2023?",
        "$383,285 million",
        "Apple's net sales were about $412 billion.",
    )
    assert v.correct is False


def test_valid_refusal_is_correct():
    v = judge(
        "How many iPhone units did Apple sell in fiscal 2025?",
        REFUSAL,
        "The filing does not disclose iPhone unit sales.",
    )
    assert v.correct is True


def test_answering_a_refusal_question_is_incorrect():
    v = judge(
        "How many iPhone units did Apple sell in fiscal 2025?",
        REFUSAL,
        "Apple sold approximately 240 million iPhones.",
    )
    assert v.correct is False
