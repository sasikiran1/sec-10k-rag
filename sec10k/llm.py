"""The LLM client. One function, `chat()`: send messages, get an answer back, and
record the call in `llm_calls` as a side effect.

Every path in the project that talks to a model goes through here, so every call
is measured and logged exactly once.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from sec10k import cache
from sec10k.config import get_settings
from sec10k.db import LlmCall, log_llm_call
from sec10k.retry import with_retries
from pydantic import BaseModel, ValidationError

# A stand-in for "whatever Pydantic model the caller asked us to fill in".
M = TypeVar("M", bound=BaseModel)

class ChatResult(BaseModel):
    """What `chat()` hands back to the caller.

    - `text`   : the model's answer (choices[0].message.content)
    - `record` : the row that was written to llm_calls (tokens, latency, ...)
    - `db_id`  : the id of that row, so callers/tests can look it up
    """

    text: str
    record: LlmCall
    db_id: int
    cached: bool = False   # True if the response came from sec10k.cache, not the API


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
        max_retries=0,  # we do our own ret/backoff in sec10k.retry
    )


def chat(
    messages: list[dict[str, str]],
    *,
    tag: str | None = None,
    response_format: dict | None = None,
) -> ChatResult:
    """Send `messages` (list of {"role", "content"}) to the active model at
    temperature 0. Serves from the response cache when possible, retries transient
    errors, writes one llm_calls row, and returns the answer + that record."""
    settings = get_settings()
    start = time.perf_counter()

    create_kwargs: dict = {
        "model": settings.active_model(),
        "temperature": 0,
        "messages": messages,
    }
    if response_format is not None:
        create_kwargs["response_format"] = response_format

    ckey = cache.key_for(create_kwargs)
    data = cache.get(ckey)
    was_cached = data is not None
    if data is None:
        resp = with_retries(lambda: _client().chat.completions.create(**create_kwargs))
        data = {
            "id": resp.id,
            "model": resp.model,
            "content": resp.choices[0].message.content,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        cache.put(ckey, data)

    latency_ms = round((time.perf_counter() - start) * 1000)
    record = LlmCall(
        provider=settings.llm_provider,
        model=data["model"],
        tag=tag,
        prompt_tokens=data["prompt_tokens"],
        completion_tokens=data["completion_tokens"],
        total_tokens=data["total_tokens"],
        cost_usd=0.0,
        latency_ms=latency_ms,
        temperature=0.0,
        response_id=data["id"],
    )
    text = data["content"]
    db_id = log_llm_call(record)
    return ChatResult(text=text, record=record, db_id=db_id, cached=was_cached)


class StructuredOutputError(RuntimeError):
    """The model never produced JSON matching the schema, even after repair attempts."""


def chat_structured(
    messages: list[dict[str, str]],
    schema: type[M],
    *,
    tag: str | None = None,
    max_repairs: int = 2,
) -> tuple[M, ChatResult]:
    """Ask the model to fill in `schema`; return (parsed_object, raw_chat_result).

    Defense 1: response_format json_object -> the reply is always valid JSON.
    Defense 2: if it's valid JSON but the wrong shape, feed the ValidationError
    back and let the model correct itself, up to `max_repairs` times.
    """
    instruction = {
        "role": "system",
        "content": (
            "Return ONE JSON object and nothing else. "
            "It must match this JSON Schema:\n" + json.dumps(schema.model_json_schema())
        ),
    }
    convo = [*messages, instruction]
    last_error: Exception | None = None

    for _ in range(max_repairs + 1):          # first try + up to max_repairs retries
        result = chat(convo, tag=tag, response_format={"type": "json_object"})
        try:
            payload = json.loads(result.text)
            return schema.model_validate(payload), result
        except (json.JSONDecodeError, ValidationError) as err:
            last_error = err
            convo = convo + [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content":
                    f"That JSON did not match the schema. Error:\n{err}\n"
                    "Reply with a corrected JSON object only."},
            ]

    raise StructuredOutputError(
        f"No schema-valid output after {max_repairs} repair attempt(s)"
    ) from last_error
