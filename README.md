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
| `chunker.py` | HTML → chunks. `chunk_fixed` = naive baseline; `chunk_structured` keeps tables whole and tags each chunk by Item |
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

See `evals/ablation.md` for the full table and caveats (small set, judge not yet
independently calibrated).

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

Ask a question; the page shows the answer and every chunk it was generated from,
with the reranker's relevance score, the filing, and the Item section.

## Tests

```
pytest -m "not live"     # no network; fast
pytest                   # includes live LLM + EDGAR calls
```
