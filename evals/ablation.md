# Ablation

Golden set: 22 hand-verified Q&A over 4 filings (Apple FY2023/FY2025, Microsoft
FY2026, NVIDIA FY2026). Generator + judge: Groq `openai/gpt-oss-120b`, temperature 0.
Retrieval metrics are over the 19 non-refusal items.

Judge calibration: κ=1.0 vs Claude-applied labels — **not** independently
human-validated (see `judge_calibration.yaml`). Treat accuracy as indicative.

| # | change | acc | recall@6 | MRR | notes |
|---|--------|-----|----------|-----|-------|
| 0 | baseline — naive vector search, k=6, unscoped | 45.5% | 26.3% | 0.113 | 11/12 misses are the model refusing for lack of context |
| 1 | + metadata filter (scope to company + fiscal year) | 54.5% | 42.1% | 0.177 | cheapest win; retrieval still misses >half the time within the right filing |
| 2 | + hybrid (RRF of vector + Postgres FTS) — **reverted** | 36.4% | 21.1% | 0.079 | regressed 18 pts. `ts_rank_cd` has no IDF weighting, so common words dominate and RRF rewards boilerplate. Code kept (`--hybrid`), not the default. |

A context-char budget was added at step 2 (prompt was overflowing Groq's 8k/min on
large table chunks). It did not change step 1's numbers — verified by re-running.

Result files: `evals/results/*.json`.
