"""Run the golden set and write a results file.

    python scripts/run_eval.py "baseline: naive retrieval" --k 6
    python scripts/run_eval.py "baseline + metadata filter" --k 6 --scoped

Writes evals/results/<timestamp>__<slug>.json and prints a summary.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from sec10k.evaluate import run_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evals" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--scoped", action="store_true", help="filter retrieval by company+fiscal_year")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between items (raise for Groq)")
    args = ap.parse_args()

    run = run_eval(args.label, k=args.k, scoped=args.scoped, sleep_between=args.sleep)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.label.lower()).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{stamp}__{slug}.json"
    out.write_text(run.model_dump_json(indent=2))

    print(f"\n{run.label}   (k={run.k}, scoped={run.scoped}, n={run.n})")
    print("-" * 60)
    print(f"answer accuracy : {run.accuracy:.1%}")
    for kind, acc in sorted(run.accuracy_by_kind.items()):
        print(f"    {kind:12} {acc:.1%}")
    print(f"recall@{run.k}       : {run.recall_at_k:.1%}   (non-refusal items)")
    print(f"MRR             : {run.mrr:.3f}")
    print(f"tokens (all)    : {run.total_tokens:,}")
    print(f"\nwrote {out}")

    misses = [i for i in run.items if not i.correct]
    if misses:
        print(f"\n{len(misses)} wrong:")
        for i in misses:
            print(f"  [{i.kind}] {i.id}: {i.predicted[:70]!r}  ({i.judge_reasoning[:60]})")


if __name__ == "__main__":
    main()
