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


def _retry_after_seconds(err: Exception) -> float | None:
    """If the server told us how long to wait (429 Retry-After header), return it."""
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    raw = resp.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()`, retrying only on RETRYABLE_ERRORS. Waits the server's
    Retry-After hint if present, else `base_delay * 2**attempt` + <=10% jitter.
    Re-raises anything non-retryable immediately, and the last error if all
    attempts fail. `sleep` is injectable so tests don't actually wait."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except RETRYABLE_ERRORS as err:
            if attempt == max_attempts - 1:
                raise
            hinted = _retry_after_seconds(err)
            if hinted is not None:
                delay = hinted + 0.25  # honor the server, plus a small cushion
            else:
                delay = base_delay * (2 ** attempt)
                delay += random.uniform(0, delay * 0.1)
            sleep(delay)
