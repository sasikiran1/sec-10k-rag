"""Seed evals/judge_calibration.yaml from the newest eval result, for hand-labeling.

    python scripts/make_calibration_set.py

Each item gets `human: "???"` — replace with "correct" or "incorrect" by reading
the question, the gold answer, and what the model actually predicted. A few
crafted borderline cases are appended to stress the judge.
"""
import glob
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "judge_calibration.yaml"

# Hand-crafted edge cases: partially-right, right-number-wrong-unit, over-refusal.
CRAFTED = [
    dict(id="crafted-partial", question="How much common stock did Apple repurchase in FY2023?",
         gold="471 million shares for $76.6 billion", predicted="471 million shares", human="???"),
    dict(id="crafted-wrong-unit", question="What were Apple's total net sales for FY2023?",
         gold="$383,285 million", predicted="$383,285", human="???"),
    dict(id="crafted-rounded", question="What was NVIDIA's FY2026 revenue?",
         gold="$215,938 million", predicted="about $216 billion", human="???"),
    dict(id="crafted-over-refuse", question="What was Microsoft's FY2026 net income?",
         gold="$133,749 million", predicted="NOT_IN_FILING", human="???"),
]


def main() -> None:
    latest = sorted(glob.glob(str(ROOT / "evals" / "results" / "*.json")))[-1]
    data = json.loads(Path(latest).read_text())

    rows = [
        dict(id=i["id"], question=i["question"], gold=i["gold"],
             predicted=i["predicted"], human="???")
        for i in data["items"]
    ] + CRAFTED

    OUT.write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {OUT} with {len(rows)} items to label (from {Path(latest).name})")


if __name__ == "__main__":
    main()
