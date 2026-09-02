"""Compare judge verdicts against human labels to see if the judge is trustworthy.

Report raw agreement AND Cohen's kappa (agreement corrected for the rate you'd
get by chance given the label distribution).
"""
from __future__ import annotations

from pydantic import BaseModel


class CalibrationReport(BaseModel):
    n: int
    agree: int
    agreement: float           # agree / n
    kappa: float               # Cohen's kappa
    disagreements: list[str]    # ids where judge != human


def score_agreement(rows: list[tuple[str, bool, bool]]) -> CalibrationReport:
    """`rows` are (id, human_correct, judge_correct). Returns raw agreement plus
    Cohen's kappa: (observed - chance) / (1 - chance), where chance agreement is
    sum over both labels of P(human=L) * P(judge=L). kappa is 1.0 when chance is 1.0."""
    n = len(rows)
    agree = sum(1 for _, h, j in rows if h == j)
    po = agree / n

    pe = 0.0
    for label in (True, False):
        p_human = sum(1 for _, h, _ in rows if h == label) / n
        p_judge = sum(1 for _, _, j in rows if j == label) / n
        pe += p_human * p_judge

    kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
    disagreements = [rid for rid, h, j in rows if h != j]

    return CalibrationReport(n=n, agree=agree, agreement=po, kappa=kappa, disagreements=disagreements)
