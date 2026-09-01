"""Agreement + Cohen's kappa math. Pure, no network.

    pytest tests/test_calibration.py -v
"""
from __future__ import annotations

from sec10k.calibration import score_agreement


def test_perfect_agreement():
    rows = [("a", True, True), ("b", False, False), ("c", True, True)]
    rep = score_agreement(rows)
    assert rep.n == 3
    assert rep.agreement == 1.0
    assert rep.kappa == 1.0
    assert rep.disagreements == []


def test_disagreements_are_listed():
    rows = [("a", True, True), ("b", True, False), ("c", False, False)]
    rep = score_agreement(rows)
    assert rep.agree == 2
    assert abs(rep.agreement - 2 / 3) < 1e-9
    assert rep.disagreements == ["b"]


def test_kappa_zero_when_agreement_is_chance():
    # human: 2 True / 2 False ; judge: 2 True / 2 False ; observed agreement 0.5
    # pe = 0.5*0.5 + 0.5*0.5 = 0.5  ->  kappa = (0.5 - 0.5) / (1 - 0.5) = 0
    rows = [("a", True, True), ("b", True, False), ("c", False, True), ("d", False, False)]
    rep = score_agreement(rows)
    assert abs(rep.agreement - 0.5) < 1e-9
    assert abs(rep.kappa - 0.0) < 1e-9


def test_kappa_one_when_all_labels_identical_and_agree():
    rows = [("a", True, True), ("b", True, True)]
    rep = score_agreement(rows)
    # pe == 1.0 (both always True) -> kappa defined as 1.0
    assert rep.kappa == 1.0
