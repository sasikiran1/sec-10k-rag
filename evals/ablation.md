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

Result files: `evals/results/*.json`.
