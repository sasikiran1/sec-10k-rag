"""Retry a callable through transient failures with exponential backoff + jitter.

Used to wrap the LLM API call. Rate limits, 5xx responses and dropped connections
are worth retrying; a malformed request or a bad key is not, and should surface
immediately.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from openai import APIConnectionError, InternalServerError, RateLimitError

T = TypeVar("T")

# "Try again in a moment" errors. Anything not in here means the request itself
# is wrong -> don't retry, let it raise.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,        # HTTP 429
    InternalServerError,   # HTTP >= 500
    APIConnectionError,    # connection dropped / timed out (APITimeoutError subclasses this)
)


def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()` and return its result, retrying on transient errors.

    - Try `fn()` up to `max_attempts` times total.
    - If it raises something that IS a RETRYABLE_ERRORS instance and attempts
      remain: wait, then try again. The wait after attempt number n (counting
      from 0) is `base_delay * (2 ** n)` seconds plus up to 10% random jitter.
    - If it raises anything else: let that exception propagate now, no retry.
    - If the final attempt also fails with a retryable error: re-raise it.
    - `sleep` is a parameter only so tests can pass a no-op instead of waiting.

    Pseudocode:
        for attempt in 0 .. max_attempts - 1:
            try:
                return fn()
            except <a RETRYABLE error> as err:
                if attempt is the last one:
                    raise
                delay = base_delay * 2**attempt, plus random jitter up to 10%
                sleep(delay)
            # a non-retryable error isn't caught here, so it propagates by itself
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except RETRYABLE_ERRORS:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            delay += random.uniform(0, delay * 0.1)
            sleep(delay)
