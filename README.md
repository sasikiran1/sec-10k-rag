# sec-10k-rag

Question answering over SEC 10-K filings, built around an **evaluation harness that
measures whether each change actually helps**. Every retrieval improvement is a
separate commit with its effect on a hand-verified question set recorded in the
message — including the changes that made things *worse*.

The RAG is ordinary. The point is the measurement: going from a number you can
defend ("naive retrieval answered 45.5% of my eval set; here is exactly which
change bought which points") instead of a vibe.

<img width="1214" height="688" alt="image" src="https://github.com/user-attachments/assets/b9a97258-3f06-402e-b702-3a463b5bbd70" />


---

## Results

22 hand-verified questions over 4 filings (Apple FY23 & FY25, Microsoft FY26,
NVIDIA FY26). Generator and judge are both Groq `openai/gpt-oss-120b` at
temperature 0. Retrieval metrics (`recall@6`, `MRR`) are over the 19 non-refusal
questions. **Each row is its own commit.**

| change | accuracy | recall@6 | MRR |
|--------|:--------:|:--------:|:---:|
| naive vector search, unscoped | 45.5% | 26.3% | 0.11 |
| + scope retrieval to the filing (company + fiscal year) | 54.5% | 42.1% | 0.18 |
| + hybrid vector · BM25 (reciprocal-rank fusion) — **regressed, reverted** | 36.4% | 21.1% | 0.08 |
| + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) | 63.6% | 57.9% | 0.23 |
| + natural-language header on each chunk before embedding | 77.3% | 73.7% | 0.49 |
| + widen the rerank candidate pool 25 → 60 | **95.5%** | **94.7%** | **0.60** |

**45.5% → 95.5%.** Notes:

- The **hybrid** row stays in the table *and* in the code (`--hybrid` flag) even
  though it lost 18 points. Deleting a failed experiment is how ablation tables
  lie. Postgres full-text `ts_rank_cd` has no IDF term weighting, so it up-ranks
  boilerplate ("Apple", "revenue"), and fusing a weak retriever with a good one
  via RRF drags the good one down.
- The **chunk header** change was the surprise: prepending one line like
  `"NVIDIA CORP — fiscal year 2026 — Item 7 (table)."` to each chunk before
  embedding recovered nearly every comparison question and roughly doubled MRR —
  terse financial tables embed poorly against natural-language questions on their
  own.
- **Wider rerank pool**: the big multi-line income-statement chunks sit around
  vector rank 30–55 (their embedding is a blur of every line item), so a top-25
  pool never showed them to the cross-encoder. Pool 60 fixed the last miss but one.

Re-validated on a harder set — 9 filings, 52 questions, adding a bank (JPMorgan),
a retailer (Walmart), and two commodity companies (ExxonMobil, Coca-Cola). The
same best config scores **82.7%**. The full ablation was *not* re-run there: on
Groq's free tier a single naive-baseline pass over 52 questions took more than five
hours of rate-limit backoff.

### Caveats you should read before trusting the numbers

- **n is small** (22, or 52). The target is 120.
- **The LLM judge is calibrated only against Claude-applied labels** (κ = 1.0,
  n = 26). That proves the judge is *self-consistent*, not that it agrees with a
  human. Independent human labelling is the top open task.
- Per-question failures and the full history are in
  [`evals/ablation.md`](evals/ablation.md).

---

## How it works

```
                       scripts/build_corpus.py  (offline, one-time)
  SEC EDGAR  ──►  fetch 10-K  ──►  structure-aware chunk  ──►  embed  ──►  Postgres
  (edgar.py)      (edgar.py)       (chunker.py)                (embeddings)   chunks table

                       answer(question)  (per request)
  question ──► vector search ──► cross-encoder rerank ──► grounded prompt ──► LLM ──► answer
              (search.py, top 60)   (rerank.py, →6)        (answer.py)       (llm.py)   + [n] cites

                       scripts/run_eval.py  (measurement)
  golden set ──► answer(q) for each ──► judge(answer, gold) ──► accuracy / recall@k / MRR
  (goldens.py)   (answer.py)             (judge.py)             (evaluate.py) → evals/results/*.json
```

**Ingestion (`ingest.py`).** A 10-K is one big HTML/iXBRL document.
`chunk_structured` walks it, renders every `<table>` to a pipe-delimited grid
kept **whole** (a blind character-window split severs the year header from its
numbers — see the `scripts/*_break.py` demos), tags each chunk with the `Item N`
section it falls under, and prepends the natural-language header. Chunks are
embedded with `all-MiniLM-L6-v2` (384-dim, local, CPU) into a `pgvector` column.

**Retrieval (`search.py`, `rerank.py`).** `answer()` embeds the question, pulls
the 60 nearest chunks by cosine distance (optionally scoped to one filing's
`company` + `fiscal_year`), then a cross-encoder
(`ms-marco-MiniLM-L-6-v2`) re-scores each `(question, chunk)` pair *together* —
which catches relevance a bi-encoder misses — and the top 6 go into the prompt.

**Generation (`answer.py`, `llm.py`).** The prompt says: answer using *only* these
excerpts; if they don't contain the answer, reply exactly `NOT_IN_FILING`. Every
model call goes through `chat()`, which caches by request hash (temperature 0 ⇒
deterministic, so re-running the eval is nearly free), retries transient errors
with exponential backoff honoring `Retry-After`, and logs tokens + latency to the
`llm_calls` table. `answer(cite=True)` (on in the web UI) asks for `[n]` markers
pointing at the excerpts used.

**Observability (`traces` table, `/traces`).** One row per `answer()` call ties
the retrieval (chunk ids + rerank scores, in final order) to the generation
(answer, refusal flag, `llm_call_id`, end-to-end latency).

**Evaluation (`evaluate.py`, `judge.py`).** For each golden question: run
`answer()`, then a separate LLM call judges the answer against the hand-verified
gold answer (numbers equal-but-formatted-differently count as a match; a valid
refusal on an unanswerable question counts as correct). Retrieval is scored
independently — a chunk is "relevant" if it contains every `must_contain`
substring the golden specifies, which is robust to re-chunking.

---

## Repository layout

Orchestration is hand-written (~1,250 non-blank lines across 18 files in
`sec10k/`), no LangChain or LlamaIndex — the point was to see every step. One
responsibility per file.

| file | responsibility |
|------|----------------|
| `config.py` | typed settings from `.env`; the only place env vars are read |
| `db.py` | Postgres connection, the `llm_calls` log, `record_trace()` |
| `cache.py` | SQLite response cache keyed on a hash of the request |
| `retry.py` | `with_retries()` — exponential backoff + jitter, honors `Retry-After` |
| `llm.py` | `chat()` (cached, retried, logged) and `chat_structured()` (JSON out + self-repair loop) |
| `embeddings.py` | text → 384-dim vector (`all-MiniLM-L6-v2`, local) |
| `search.py` | `search()` vector · `keyword_search()` Postgres FTS · `hybrid_search()` RRF |
| `rerank.py` | `rerank()` — cross-encoder re-scores a candidate pool |
| `edgar.py` | fetch a 10-K from SEC EDGAR (ticker → CIK → primary document) |
| `chunker.py` | HTML → chunks; tables kept whole, tagged by `Item` section |
| `ingest.py` | edgar + chunker + embeddings → DB, with a NL header per chunk |
| `answer.py` | `answer(question)` — retrieve → grounded prompt → `chat()`. The product. |
| `goldens.py` | load `evals/goldens.yaml` (hand-verified Q&A) |
| `judge.py` | LLM decides whether an answer matches the gold answer |
| `evaluate.py` | `run_eval()` — loop goldens → answer → judge → accuracy / recall@k / MRR |
| `calibration.py` | judge-vs-human agreement + Cohen's kappa |
| `app.py` | FastAPI demo: `/ask`, `/filings`, `/traces` |

`scripts/` are runnable entrypoints (never imported). `tests/` mirrors `sec10k/`
file-for-file; `pytest -m "not live"` needs only Postgres, `pytest` also makes
real LLM + EDGAR calls. `evals/` holds the golden set, the ablation write-up, and
one JSON per eval run. `web/index.html` is the single-file front end.

---

## Setup

**Prerequisites:** Python 3.12+, Docker, and a free
[Groq API key](https://console.groq.com). Developed on WSL2 (Ubuntu) — the ML
stack (`torch`, `tokenizers`, `lxml`) is unsigned native code that Windows Smart
App Control blocks intermittently, so a Linux environment is smoother.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
#   GROQ_API_KEY   = your key
#   SEC_USER_AGENT = "your-name your@email.com"   (SEC rejects requests without one)

docker compose up -d        # Postgres 16 + pgvector on localhost:5433 (db/user/pass all "sec10k")
python scripts/build_corpus.py     # fetches 9 10-Ks, chunks, embeds → ~3,300 rows (a few minutes, no LLM calls)
```

---

## Usage

### Web demo

```bash
uvicorn sec10k.app:app        # http://localhost:8000
```

Pick a filing (or "any filing"), ask a question. The page shows the answer with
clickable `[n]` citations, then the six chunks it was built from — each with the
cross-encoder's relevance score, the filing, and the `Item` section.
`http://localhost:8000/traces` lists every question asked with its retrieval.

### Run the evaluation

```bash
# naive baseline
python scripts/run_eval.py "baseline" --sleep 18

# the best config
python scripts/run_eval.py "scoped + rerank" --scoped --rerank --sleep 18
```

`--sleep 18` paces requests under Groq's free-tier limit (~8k tokens/minute).
Results print a summary and write `evals/results/<timestamp>__<label>.json`.
Re-runs with an unchanged prompt/context are served from the cache and finish in
seconds.

### Calibrate the judge

```bash
python scripts/calibrate_judge.py      # judge vs the labels in evals/judge_calibration.yaml
```

### Tests

```bash
pytest -m "not live"     # no network; needs Postgres up
pytest                   # also makes live LLM + EDGAR calls
```

---

## Stack

Python 3.12+ · Postgres 16 + `pgvector` · `sentence-transformers` (local, CPU) ·
Groq free tier via the OpenAI-compatible API · FastAPI · pytest.
