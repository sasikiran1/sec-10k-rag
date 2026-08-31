"""The LLM client. One function, `chat()`: send messages, get an answer back, and
record the call in `llm_calls` as a side effect.

Every path in the project that talks to a model goes through here, so every call
is measured and logged exactly once.
"""
from __future__ import annotations

import time
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from sec10k.config import get_settings
from sec10k.db import LlmCall, log_llm_call


class ChatResult(BaseModel):
    """What `chat()` hands back to the caller.

    - `text`   : the model's answer (choices[0].message.content)
    - `record` : the row that was written to llm_calls (tokens, latency, ...)
    - `db_id`  : the id of that row, so callers/tests can look it up
    """

    text: str
    record: LlmCall
    db_id: int


@lru_cache
def _client() -> OpenAI:
    """Build the OpenAI-compatible client once, pointed at the active provider.

    lru_cache => one client object reused for the whole process instead of a new
    one (and new connection pool) per call.
    """
    settings = get_settings()
    return OpenAI(
        api_key=settings.active_api_key(),
        base_url=settings.active_base_url(),
    )


def chat(messages: list[dict[str, str]], *, tag: str | None = None) -> ChatResult:
    """Send `messages` to the active model at temperature 0, log the call, return it.

    `messages` is a list of {"role": ..., "content": ...} dicts, exactly the shape
    the API wants (roles: "system", "user", "assistant").

    Steps to implement:
      1. settings = get_settings()
      2. Read the clock:  start = time.perf_counter()
      3. resp = _client().chat.completions.create(
             model=settings.active_model(),
             temperature=0,
             messages=messages,
         )
      4. latency_ms = round((time.perf_counter() - start) * 1000)
      5. Pull out:
           text  = resp.choices[0].message.content
           usage = resp.usage
      6. Build an LlmCall(...) with:
           provider=settings.llm_provider,
           model=resp.model,            # what the server actually used
           tag=tag,
           prompt_tokens=usage.prompt_tokens,
           completion_tokens=usage.completion_tokens,
           total_tokens=usage.total_tokens,
           cost_usd=0.0,                # free tier for now; real rates come later
           latency_ms=latency_ms,
           temperature=0.0,
           response_id=resp.id,
      7. db_id = log_llm_call(record)
      8. return ChatResult(text=text, record=record, db_id=db_id)
    """
    settings = get_settings()
    start = time.perf_counter()
    resp = _client().chat.completions.create(
        model=settings.active_model(),
        temperature=0,
        messages=messages,
    )
    latency_ms = round((time.perf_counter() - start) * 1000)
    text = resp.choices[0].message.content
    usage = resp.usage
    record = LlmCall(
        provider=settings.llm_provider,
        model=resp.model,
        tag=tag,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=0.0,
        latency_ms=latency_ms,
        temperature=0.0,
        response_id=resp.id,
    )
    db_id = log_llm_call(record)
    return ChatResult(text=text, record=record, db_id=db_id)
