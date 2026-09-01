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
    """rows are (id, human_correct, judge_correct).

    - agreement = fraction of rows where human_correct == judge_correct
    - kappa:
        po = agreement (observed)
        For each label L in {True, False}:
            p_human_L = fraction of rows where human_correct == L
            p_judge_L = fraction of rows where judge_correct == L
        pe = sum over L of (p_human_L * p_judge_L)          # chance agreement
        kappa = (po - pe) / (1 - pe)     (define kappa = 1.0 if pe == 1.0)
    - disagreements = [id for rows where human != judge]
    """
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
