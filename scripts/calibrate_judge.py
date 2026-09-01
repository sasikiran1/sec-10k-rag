"""Run the judge on the hand-labeled calibration set and report agreement.

    python scripts/calibrate_judge.py

Reads evals/judge_calibration.yaml (every `human` must be "correct"/"incorrect"),
runs judge() on each, prints raw agreement, Cohen's kappa, and every disagreement.
"""
from pathlib import Path

import yaml

from sec10k.calibration import score_agreement
from sec10k.judge import judge

PATH = Path(__file__).resolve().parent.parent / "evals" / "judge_calibration.yaml"


def main() -> None:
    rows = yaml.safe_load(PATH.read_text())

    unlabeled = [r["id"] for r in rows if r["human"] not in ("correct", "incorrect")]
    if unlabeled:
        raise SystemExit(f"label these first (correct/incorrect): {unlabeled}")

    triples = []
    details = {}
    for r in rows:
        verdict, _ = judge(r["question"], r["gold"], r["predicted"], tag=f"calib:{r['id']}")
        human = r["human"] == "correct"
        triples.append((r["id"], human, verdict.correct))
        details[r["id"]] = (human, verdict.correct, verdict.reasoning)

    rep = score_agreement(triples)
    print(f"\ncalibration: n={rep.n}  agreement={rep.agreement:.1%}  kappa={rep.kappa:.3f}")
    if rep.disagreements:
        print("\ndisagreements (human -> judge):")
        for rid in rep.disagreements:
            h, j, why = details[rid]
            print(f"  {rid}: human={'correct' if h else 'incorrect'}  "
                  f"judge={'correct' if j else 'incorrect'}  ({why[:80]})")
    verdict_word = "trust it" if rep.kappa >= 0.8 else "fix the rubric" if rep.kappa < 0.6 else "borderline"
    print(f"\n-> {verdict_word}")


if __name__ == "__main__":
    main()
