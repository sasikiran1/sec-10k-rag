"""Defines what with_retries() must do. No network — all failures are faked.

    pytest tests/test_retry.py -v
"""
from __future__ import annotations

import httpx2 as httpx  # the HTTP lib the openai SDK (v3) is built on
import pytest
from openai import APIConnectionError

from sec10k.retry import with_retries

NO_SLEEP = lambda _delay: None  # noqa: E731  (fine for a test helper)


def _conn_error() -> APIConnectionError:
    """A retryable error we can raise without touching the network."""
    return APIConnectionError(request=httpx.Request("POST", "https://example.test"))


def test_returns_result_on_first_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert with_retries(fn, sleep=NO_SLEEP) == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _conn_error()
        return "recovered"

    assert with_retries(fn, max_attempts=4, sleep=NO_SLEEP) == "recovered"
    assert len(calls) == 3


def test_gives_up_after_max_attempts_and_reraises():
    calls = []

    def fn():
        calls.append(1)
        raise _conn_error()

    with pytest.raises(APIConnectionError):
        with_retries(fn, max_attempts=3, sleep=NO_SLEEP)
    assert len(calls) == 3


def test_non_retryable_error_propagates_immediately():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        with_retries(fn, max_attempts=5, sleep=NO_SLEEP)
    assert len(calls) == 1  # not retried


def test_backoff_delays_grow_and_there_is_one_per_gap():
    slept: list[float] = []

    def fn():
        raise _conn_error()

    with pytest.raises(APIConnectionError):
        with_retries(fn, max_attempts=4, base_delay=1.0, sleep=slept.append)

    assert len(slept) == 3                       # 3 waits between 4 attempts
    assert slept[0] < slept[1] < slept[2]        # each wait longer than the last
    assert slept[0] >= 1.0 and slept[1] >= 2.0 and slept[2] >= 4.0
    assert slept[0] <= 1.1 and slept[1] <= 2.2 and slept[2] <= 4.4  # jitter <= 10%
