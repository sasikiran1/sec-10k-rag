"""What chat_structured() must do. The deterministic tests stub out chat() so the
repair loop runs offline; one live test hits the real model.

    pytest tests/test_structured.py -v
    pytest tests/test_structured.py -v -m "not live"
"""
from __future__ import annotations

import psycopg
import pytest
from pydantic import BaseModel

from sec10k.config import get_settings
from sec10k.db import LlmCall
from sec10k.llm import ChatResult, StructuredOutputError, chat_structured


class Summary(BaseModel):
    item: str
    title: str


def _fake_result(text: str) -> ChatResult:
    """A ChatResult with `text` set and dummy metadata — enough for the loop."""
    return ChatResult(
        text=text,
        record=LlmCall(
            provider="fake", model="fake", tag=None,
            prompt_tokens=1, completion_tokens=1, total_tokens=2,
            cost_usd=0.0, latency_ms=1, temperature=0.0, response_id="fake",
        ),
        db_id=0,
    )


def _stub_chat(monkeypatch, responses: list[ChatResult]):
    """Make sec10k.llm.chat hand back `responses` in order; record the calls."""
    calls: list[list[dict]] = []

    def fake_chat(messages, *, tag=None, response_format=None):
        calls.append(messages)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr("sec10k.llm.chat", fake_chat)
    return calls


def test_valid_first_response_returns_without_repair(monkeypatch):
    calls = _stub_chat(monkeypatch, [_fake_result('{"item": "Item 7", "title": "MD&A"}')])

    obj, result = chat_structured([{"role": "user", "content": "x"}], Summary)

    assert isinstance(obj, Summary)
    assert obj.item == "Item 7"
    assert len(calls) == 1


def test_bad_shape_is_repaired(monkeypatch):
    calls = _stub_chat(monkeypatch, [
        _fake_result('{"item": "Item 7"}'),                     # missing 'title'
        _fake_result('{"item": "Item 7", "title": "MD&A"}'),    # corrected
    ])

    obj, result = chat_structured([{"role": "user", "content": "x"}], Summary, max_repairs=2)

    assert obj.title == "MD&A"
    assert len(calls) == 2

    # the 2nd call must carry the model's bad answer + our complaint
    repair_msgs = calls[1]
    assert [m["role"] for m in repair_msgs].count("assistant") == 1
    assert any(
        m["role"] == "user" and "did not match the schema" in m["content"]
        for m in repair_msgs
    )


def test_non_json_text_is_repaired(monkeypatch):
    calls = _stub_chat(monkeypatch, [
        _fake_result('here you go: {"item": "7"}'),             # not parseable
        _fake_result('{"item": "Item 7", "title": "MD&A"}'),
    ])

    obj, _ = chat_structured([{"role": "user", "content": "x"}], Summary)

    assert obj.title == "MD&A"
    assert len(calls) == 2


def test_raises_after_exhausting_repairs(monkeypatch):
    calls = _stub_chat(monkeypatch, [_fake_result('{"item": "Item 7"}')])  # always broken

    with pytest.raises(StructuredOutputError):
        chat_structured([{"role": "user", "content": "x"}], Summary, max_repairs=2)

    assert len(calls) == 3  # first try + 2 repairs


@pytest.mark.live
def test_chat_structured_live():
    obj, result = chat_structured(
        [{"role": "user", "content": "Describe Item 1A of a Form 10-K."}],
        Summary,
        tag="pytest-structured",
    )
    try:
        assert isinstance(obj, Summary)
        assert obj.item.strip() and obj.title.strip()
    finally:
        with psycopg.connect(get_settings().database_url) as conn:
            conn.execute("DELETE FROM llm_calls WHERE tag = %s", ("pytest-structured",))
