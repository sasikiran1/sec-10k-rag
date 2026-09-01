"""Cache key stability + get/put round-trip. No network.

    pytest tests/test_cache.py -v
"""
from __future__ import annotations

import sec10k.cache as cache


def test_key_is_order_independent():
    a = cache.key_for({"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hi"}]})
    b = cache.key_for({"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "model": "m"})
    assert a == b


def test_key_changes_with_content():
    a = cache.key_for({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    b = cache.key_for({"model": "m", "messages": [{"role": "user", "content": "hello"}]})
    assert a != b


def test_put_then_get_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_DB", tmp_path / "llm.sqlite")
    cache.put("k1", {"choices": [{"message": {"content": "x"}}]})
    assert cache.get("k1") == {"choices": [{"message": {"content": "x"}}]}
    assert cache.get("missing") is None


def test_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_DB", tmp_path / "llm.sqlite")
    monkeypatch.setenv("LLM_CACHE", "0")
    cache.put("k", {"a": 1})
    assert cache.get("k") is None  # writes and reads are no-ops when disabled
