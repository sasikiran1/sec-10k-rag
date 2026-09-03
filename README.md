# sec-10k-rag

Question answering over SEC 10-K filings, with an evaluation harness that measures
whether each retrieval change actually helps.

The interesting part isn't the RAG — it's `evals/ablation.md`, where every change
is a separate commit with its measured effect on a hand-verified question set
(including the changes that made things worse).

## Architecture

Orchestration is hand-written (~1,300 lines, `sec10k/`), no LangChain/LlamaIndex —
the point was to see every step. One responsibility per file.

**Foundations** — no dependencies on the rest of the code
| `config.py` | typed settings from `.env`; everything reads config here |
| `db.py` | Postgres connection + the `llm_calls` cost/latency log |
| `cache.py` | SQLite response cache, keyed on a hash of the request (temp 0 ⇒ deterministic) |
| `retry.py` | `with_retries()` — exponential backoff, honors `Retry-After` |

**LLM layer** — the only door to a model
| `llm.py` | `chat()` (cached, retried, logged) and `chat_structured()` (JSON out + self-repair loop) |

**Retrieval**
| `embeddings.py` | text → 384-dim vector (local `all-MiniLM-L6-v2`) |
| `search.py` | `search()` vector · `keyword_search()` Postgres FTS · `hybrid_search()` RRF fusion |
| `rerank.py` | `rerank()` — cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool |

**Ingestion** — offline, builds the `chunks` table
| `edgar.py` | fetch a 10-K from SEC EDGAR (ticker → CIK → primary document) |
| `chunker.py` | HTML → chunks. `chunk_structured` keeps tables whole and tags each chunk by Item section |
| `ingest.py` | glue: edgar + chunker + embeddings → DB, with a natural-language header per chunk |

**QA + evaluation**
| `answer.py` | `answer(question)` = retrieve → grounded prompt → `chat()`. The product. |
| `goldens.py` | load `evals/goldens.yaml` (hand-verified Q&A) |
| `judge.py` | LLM decides whether an answer matches the gold answer |
| `evaluate.py` | `run_eval()` = loop goldens → answer → judge → accuracy / recall@k / MRR |
| `calibration.py` | judge-vs-human agreement + Cohen's kappa |

`scripts/` are runnable entrypoints (never imported); `tests/` mirrors `sec10k/`
file-for-file. `*_break.py` scripts reproduce specific retrieval failures.

Flow: `build_corpus.py` fills `chunks` → `answer("…")` retrieves + reranks + calls
`chat()` → `run_eval.py` scores it against the golden set.

## Results

22 hand-verified questions over 4 filings (Apple FY23/FY25, Microsoft FY26, NVIDIA
FY26). Generator + judge: Groq `openai/gpt-oss-120b`, temperature 0. Each row is a
separate commit; retrieval metrics are over the 19 non-refusal items.

| change | accuracy | recall@6 | MRR |
|--------|:--------:|:--------:|:---:|
| naive vector search, unscoped | 45.5% | 26.3% | 0.11 |
| + scope retrieval to the filing (company + fiscal year) | 54.5% | 42.1% | 0.18 |
| + hybrid vector·BM25 (RRF) — **regressed, reverted** | 36.4% | 21.1% | 0.08 |
| + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) | 63.6% | 57.9% | 0.23 |
| + natural-language header on each chunk before embedding | 77.3% | 73.7% | 0.49 |
| + widen rerank candidate pool 25 → 60 | **95.5%** | **94.7%** | **0.60** |

**45.5% → 95.5%.** The hybrid step is kept in the table and the code (`--hybrid`)
because deleting a failed experiment is how ablation tables lie — Postgres FTS has
no IDF weighting, so it up-ranks boilerplate and RRF drags the good retriever down.

Re-validated on a harder set (9 filings, 52 questions incl. a bank and two
commodity companies): the same config scores **82.7%** — the full ablation wasn't
re-run there (a single free-tier baseline pass = 5h of rate-limit backoff).

**Caveats.** n is small. The LLM judge is calibrated only against Claude-applied
labels (κ=1.0), not an independent human rater — so read the numbers as "strong on
a small set with a self-consistent judge." Full detail and per-question failures in
`evals/ablation.md`.

## Stack

Python 3.12+ · Postgres 16 + pgvector · sentence-transformers (local, CPU) ·
Groq / Gemini free tiers via the OpenAI-compatible API · pytest.

## Dev setup

Developed on WSL2 (Ubuntu) — the ML stack (torch, tokenizers, lxml) is unsigned
native code that Windows Smart App Control blocks intermittently.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env      # then fill in GROQ_API_KEY and SEC_USER_AGENT
docker compose up -d
python scripts/build_corpus.py
```

Postgres listens on `localhost:5433` (db/user/pass all `sec10k`).

## Demo UI

```
uvicorn sec10k.app:app --reload    # http://localhost:8000
```

Ask a question; the page shows the answer (with clickable `[n]` citations into the
sources) and every chunk it was generated from — the reranker's relevance score,
the filing, and the Item section.

## Tests

```
pytest -m "not live"     # no network; fast
pytest                   # includes live LLM + EDGAR calls
```
