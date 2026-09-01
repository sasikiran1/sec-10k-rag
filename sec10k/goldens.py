"""The evaluation golden set: hand-written, human-verified question/answer pairs.

Loaded from evals/goldens.yaml. Every `answer` here was checked against the real
filing text — nothing in this file is generated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

GOLDENS_PATH = Path(__file__).resolve().parent.parent / "evals" / "goldens.yaml"

REFUSAL = "NOT_IN_FILING"  # gold answer for questions the filing can't answer


class Golden(BaseModel):
    """One evaluation item."""

    id: str
    question: str
    answer: str                       # gold answer, or REFUSAL
    company: str                      # matches chunks.company
    fiscal_year: int                  # matches chunks.fiscal_year
    kind: Literal["single", "multi_hop", "comparison", "refusal"]
    # Substrings that a chunk must ALL contain to count as "relevant" for this
    # question, used to score retrieval without pinning volatile chunk ids.
    must_contain: list[str] = []
    note: str = ""


def load_goldens(path: Path | str = GOLDENS_PATH) -> list[Golden]:
    """Parse evals/goldens.yaml into a list of Golden, validating each item.

        raw = yaml.safe_load(Path(path).read_text())
        return [Golden(**item) for item in raw]
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Golden(**item) for item in raw]
