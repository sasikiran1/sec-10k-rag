"""Answer a question from retrieved 10-K chunks.

retrieve -> stuff the chunks into a grounded prompt -> generate. Deliberately
plain: this is the baseline the retrieval improvements in session 6 must beat.
"""
from __future__ import annotations

import time

from pydantic import BaseModel

from sec10k.db import record_trace
from sec10k.goldens import REFUSAL
from sec10k.llm import ChatResult, chat
from sec10k.rerank import rerank as rerank_hits
from sec10k.search import Hit, hybrid_search, search

SYSTEM_PROMPT = (
    "You answer questions about SEC 10-K filings using ONLY the provided excerpts. "
    f"If the excerpts do not contain the answer, reply with exactly: {REFUSAL}\n"
    "Be concise: give the figure or fact asked for, with its unit. No commentary."
)

# Appended to SYSTEM_PROMPT when cite=True. Kept out of the default (measured)
# prompt so turning citations on doesn't invalidate the ablation numbers.
CITE_CLAUSE = (
    " After the answer, cite the excerpt number(s) you used in square brackets, "
    "e.g. [2] or [1][3]. Cite only excerpts that actually contain the answer."
)


class AnswerResult(BaseModel):
    question: str
    answer: str
    hits: list[Hit]
    chat: ChatResult      # carries tokens / latency / db_id for cost accounting


CONTEXT_CHAR_BUDGET = 16000  # ~4k tokens; keeps the request under Groq's 8k/min


def format_context(hits: list[Hit], *, char_budget: int = CONTEXT_CHAR_BUDGET) -> str:
    """Render hits as a numbered context block, best-first, within a char budget.

    Adds whole chunks in rank order until the next one wouldn't fit, then stops —
    so tables stay intact and the prompt stays bounded. Always includes at least
    the top hit.
    """
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits):
        entry = f"[{i + 1}] ({h.company} FY{h.fiscal_year}, {h.section})\n{h.text}"
        if parts and used + len(entry) > char_budget:
            break
        if not parts and len(entry) > char_budget:
            entry = entry[:char_budget] + " …[truncated]"
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


def answer(
    question: str,
    *,
    k: int = 6,
    tag: str = "answer",
    company: str | None = None,
    fiscal_year: int | None = None,
    hybrid: bool = False,
    rerank: bool = True,
    rerank_pool: int = 60,
    max_retries: int = 6,
    cite: bool = False,
    trace: bool = True,
) -> AnswerResult:
    """Retrieve chunks for `question` (vector, optionally hybrid, optionally
    cross-encoder reranked from a `rerank_pool`), build a grounded prompt, and
    generate. `company` / `fiscal_year` scope retrieval to one filing.
    `cite=True` asks the model to append [n] references to the excerpts it used
    (opt-in — the eval runs without it). Interactive callers should pass a small
    `max_retries` (e.g. 2) so a rate-limited call fails fast."""
    started = time.perf_counter()
    retrieve = hybrid_search if hybrid else search
    if rerank:
        pool = retrieve(question, k=rerank_pool, company=company, fiscal_year=fiscal_year)
        hits = rerank_hits(question, pool, top_k=k)
    else:
        hits = retrieve(question, k=k, company=company, fiscal_year=fiscal_year)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + (CITE_CLAUSE if cite else "")},
        {
            "role": "user",
            "content": f"Excerpts:\n{format_context(hits)}\n\nQuestion: {question}",
        },
    ]
    result = chat(messages, tag=tag, max_retries=max_retries)
    text = result.text.strip()

    if trace:
        record_trace(
            question=question, company=company, fiscal_year=fiscal_year,
            hybrid=hybrid, reranked=rerank,
            retrieved=[
                {"chunk_id": h.id, "score": round(h.score, 4),
                 "section": h.section, "kind": h.kind}
                for h in hits
            ],
            answer=text, refused=REFUSAL in text, llm_call_id=result.db_id,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    return AnswerResult(question=question, answer=text, hits=hits, chat=result)
