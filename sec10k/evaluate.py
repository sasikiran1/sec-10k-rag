"""Run the golden set through answer() + judge() and roll up the numbers.

Two families of metric:
  - answer accuracy  : did judge() call the answer correct? (overall + by kind)
  - retrieval quality : recall@k and MRR over the non-refusal items, using each
                        golden's `must_contain` strings to mark chunks relevant.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from pydantic import BaseModel

from sec10k.answer import answer
from sec10k.goldens import Golden, load_goldens
from sec10k.judge import judge
from sec10k.search import Hit


class ItemResult(BaseModel):
    id: str
    kind: str
    question: str
    gold: str
    predicted: str
    correct: bool
    score: float
    judge_reasoning: str
    retrieved: list[int]                 # chunk ids, best first
    relevant_ranks: list[int]            # 1-indexed ranks that were relevant
    recall_at_k: float | None            # None for refusal items
    reciprocal_rank: float | None
    answer_tokens: int
    judge_tokens: int


class EvalRun(BaseModel):
    label: str
    k: int
    scoped: bool
    n: int
    timestamp: str
    accuracy: float
    accuracy_by_kind: dict[str, float]
    recall_at_k: float
    mrr: float
    total_tokens: int
    items: list[ItemResult]


# --- metrics you implement --------------------------------------------------

def is_relevant(hit: Hit, golden: Golden) -> bool:
    """True iff `hit.text` contains every string in `golden.must_contain`."""
    return all(s in hit.text for s in golden.must_contain)


def score_retrieval(hits: list[Hit], golden: Golden) -> tuple[list[int], float, float]:
    """Score one question's retrieval.

    Returns (relevant_ranks, recall_at_k, reciprocal_rank):
      - relevant_ranks : 1-indexed positions in `hits` where is_relevant() is True
      - recall_at_k    : 1.0 if relevant_ranks is non-empty else 0.0
      - reciprocal_rank: 1 / first relevant rank, else 0.0
    """
    ranks = [i for i, h in enumerate(hits, start=1) if is_relevant(h, golden)]
    recall = 1.0 if ranks else 0.0
    rr = 1.0 / ranks[0] if ranks else 0.0
    return ranks, recall, rr


def aggregate(label: str, k: int, scoped: bool, items: list[ItemResult]) -> EvalRun:
    """Roll ItemResults into an EvalRun.

      - accuracy         : fraction of items with correct == True
      - accuracy_by_kind : same, grouped by item.kind
      - recall_at_k, mrr : mean over items whose recall_at_k is not None
      - total_tokens     : sum of answer_tokens + judge_tokens
      - timestamp        : datetime.now(timezone.utc).isoformat(timespec="seconds")
    """
    n = len(items)
    accuracy = sum(i.correct for i in items) / n if n else 0.0

    by_kind: dict[str, float] = {}
    for kind in {i.kind for i in items}:
        group = [i for i in items if i.kind == kind]
        by_kind[kind] = sum(i.correct for i in group) / len(group)

    scored = [i for i in items if i.recall_at_k is not None]
    recall_at_k = sum(i.recall_at_k for i in scored) / len(scored) if scored else 0.0
    mrr = sum(i.reciprocal_rank for i in scored) / len(scored) if scored else 0.0

    return EvalRun(
        label=label, k=k, scoped=scoped, n=n,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        accuracy=accuracy, accuracy_by_kind=by_kind,
        recall_at_k=recall_at_k, mrr=mrr,
        total_tokens=sum(i.answer_tokens + i.judge_tokens for i in items),
        items=items,
    )


# --- orchestration (already wired) ----------------------------------------

def evaluate_item(golden: Golden, *, k: int, scoped: bool) -> ItemResult:
    company = golden.company if scoped else None
    fiscal_year = golden.fiscal_year if scoped else None

    res = answer(
        golden.question, k=k, tag=f"eval:{golden.id}:answer",
        company=company, fiscal_year=fiscal_year,
    )
    verdict, judge_chat = judge(
        golden.question, golden.answer, res.answer, tag=f"eval:{golden.id}:judge"
    )

    if golden.kind == "refusal":
        ranks, recall, rr = [], None, None
    else:
        ranks, recall, rr = score_retrieval(res.hits, golden)

    return ItemResult(
        id=golden.id, kind=golden.kind, question=golden.question,
        gold=golden.answer, predicted=res.answer,
        correct=verdict.correct, score=verdict.score,
        judge_reasoning=verdict.reasoning,
        retrieved=[h.id for h in res.hits], relevant_ranks=ranks,
        recall_at_k=recall, reciprocal_rank=rr,
        answer_tokens=res.chat.record.total_tokens,
        judge_tokens=judge_chat.record.total_tokens,
    )


def run_eval(
    label: str, *, k: int = 6, scoped: bool = False, sleep_between: float = 1.0
) -> EvalRun:
    """Evaluate every golden. `sleep_between` paces the loop to stay under the
    provider's rate limit (Groq free tier is ~8k tokens/min; raise it there)."""
    goldens = load_goldens()
    items: list[ItemResult] = []
    for n, g in enumerate(goldens):
        items.append(evaluate_item(g, k=k, scoped=scoped))
        if sleep_between and n < len(goldens) - 1:
            time.sleep(sleep_between)
    return aggregate(label, k, scoped, items)
