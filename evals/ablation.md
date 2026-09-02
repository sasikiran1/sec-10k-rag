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
| 3 | + cross-encoder reranker (ms-marco-MiniLM-L-6-v2, top-25 → 6) | 63.6% | 57.9% | 0.230 | +9 acc, +16 recall. Comparison acc fell 67→33: the CE prefers question-phrase matches over the multi-year table. 5 of 8 misses were NVIDIA. |
| 4 | + natural-language chunk header before embedding (`"<company> — fiscal year <yr> — <section> (<kind>)."`) | **77.3%** | **73.7%** | **0.493** | +14 acc, +16 recall, MRR ~2×. Terse tables now embed/rerank against NL queries; recovered comparison (→67) and 4 of 5 NVIDIA. 5 misses left (Apple FY25 + MSFT revenue chunks, 1 NVDA figure disambiguation). |

Steps 0–3 are on the pre-enrichment corpus; step 4 re-ingests with the header and
re-runs step 3's exact config, so the 3→4 delta isolates the enrichment. A
context-char budget was added at step 2 (prompt was overflowing Groq's 8k/min on
large table chunks); it did not change step 1's numbers — verified by re-running.

Result files: `evals/results/*.json`.
