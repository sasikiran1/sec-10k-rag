"""Answer a question from retrieved 10-K chunks.

retrieve -> stuff the chunks into a grounded prompt -> generate. Deliberately
plain: this is the baseline the retrieval improvements in session 6 must beat.
"""
from __future__ import annotations

from pydantic import BaseModel

from sec10k.goldens import REFUSAL
from sec10k.llm import ChatResult, chat
from sec10k.search import Hit, search

SYSTEM_PROMPT = (
    "You answer questions about SEC 10-K filings using ONLY the provided excerpts. "
    f"If the excerpts do not contain the answer, reply with exactly: {REFUSAL}\n"
    "Be concise: give the figure or fact asked for, with its unit. No commentary."
)


class AnswerResult(BaseModel):
    question: str
    answer: str
    hits: list[Hit]
    chat: ChatResult      # carries tokens / latency / db_id for cost accounting


def format_context(hits: list[Hit]) -> str:
    """Render hits as a numbered context block for the prompt.

    One entry per hit:
        [1] (Apple Inc. FY2025, Item 8)
        <chunk text>

        [2] (...)
        ...
    Join entries with a blank line.
    """
    return "\n\n".join(
        f"[{i + 1}] ({h.company} FY{h.fiscal_year}, {h.section})\n{h.text}"
        for i, h in enumerate(hits)
    )


def answer(question: str, *, k: int = 6, tag: str = "answer") -> AnswerResult:
    """Retrieve k chunks for `question`, generate a grounded answer, return both.

    Steps:
      1. hits = search(question, k=k)
      2. messages = [
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content":
                 f"Excerpts:\n{format_context(hits)}\n\nQuestion: {question}"},
         ]
      3. result = chat(messages, tag=tag)
      4. return AnswerResult(question=question, answer=result.text.strip(),
                             hits=hits, chat=result)
    """
    hits = search(question, k=k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Excerpts:\n{format_context(hits)}\n\nQuestion: {question}",
        },
    ]
    result = chat(messages, tag=tag)
    return AnswerResult(
        question=question, answer=result.text.strip(), hits=hits, chat=result
    )
