"""Tests that define what `Settings` must do. Implement config.py until these pass:

    pytest tests/test_config.py -v
"""
from __future__ import annotations

import pytest

from sec10k.config import Settings


def test_defaults_pick_groq():
    s = Settings(_env_file=None)  # ignore any real .env for this test
    assert s.llm_provider == "groq"
    assert s.active_base_url() == "https://api.groq.com/openai/v1"
    assert s.active_model() == "openai/gpt-oss-120b"


def test_active_key_follows_provider():
    s = Settings(
        _env_file=None,
        llm_provider="gemini",
        groq_api_key="groq-key",
        gemini_api_key="gemini-key",
    )
    assert s.active_api_key() == "gemini-key"
    assert s.active_base_url() == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert s.active_model() == "gemini-2.0-flash"


def test_missing_key_for_active_provider_raises():
    s = Settings(_env_file=None, llm_provider="groq", groq_api_key="")
    with pytest.raises(RuntimeError):
        s.active_api_key()


def test_unknown_provider_rejected_by_type():
    # Literal type means pydantic should reject this at construction.
    with pytest.raises(Exception):
        Settings(_env_file=None, llm_provider="openai")
