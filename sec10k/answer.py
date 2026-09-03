"""Generate grounded answers from retrieved SEC 10-K chunks.

The generation layer is intentionally simple so retrieval changes can be
evaluated independently.
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

# Citation mode is opt-in so the measured evaluation prompt remains unchanged.
CITE_CLAUSE = (
    " After the answer, cite the excerpt number(s) you used in square brackets, "
    "e.g. [2] or [1][3]. Cite only excerpts that actually contain the answer."
)


class AnswerResult(BaseModel):
    question: str
    answer: str
    hits: list[Hit]
    chat: ChatResult


CONTEXT_CHAR_BUDGET = 16000  # ~4k tokens; keeps the request under Groq's 8k/min


def format_context(hits: list[Hit], *, char_budget: int = CONTEXT_CHAR_BUDGET) -> str:
    """Render ranked hits as a bounded, numbered context block."""
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
    """Retrieve evidence, optionally rerank it, and generate a grounded answer.

    `company` and `fiscal_year` scope retrieval to one filing. `cite=True` adds
    excerpt references without changing the default evaluation prompt.
    """
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
