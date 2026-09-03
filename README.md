# Filing Analyst

### Evaluation-driven RAG for SEC 10-K filings

> **45.5% → 95.5% answer accuracy through measured retrieval improvements — then 82.7% on a harder 52-question / 9-filing validation set.**

Question answering over SEC 10-K filings, built around an **evaluation harness that measures whether each change actually helps**.

Every retrieval improvement is isolated, evaluated, and recorded — including changes that made performance **worse**.

The RAG itself is intentionally straightforward. The point is the measurement:

> *“Naive retrieval answered 45.5% of my eval set. Here is exactly which change bought which points.”*

https://github.com/user-attachments/assets/741f2984-3ba0-4afb-b172-4467cbd7e1ae

---

## Results

### Initial ablation

22 hand-verified questions over 4 filings:

- Apple FY2023
- Apple FY2025
- Microsoft FY2026
- NVIDIA FY2026

Generator and judge are both Groq `openai/gpt-oss-120b` at temperature `0`.

Retrieval metrics (`Recall@6`, `MRR`) are computed over the 19 non-refusal questions.

**Each row represents a separate experiment and commit.**

| Change | Accuracy | Recall@6 | MRR |
|---|:---:|:---:|:---:|
| Naive vector search, unscoped | 45.5% | 26.3% | 0.11 |
| + scope retrieval to company + fiscal year | 54.5% | 42.1% | 0.18 |
| + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) | 63.6% | 57.9% | 0.23 |
| + natural-language header before embedding | 77.3% | 73.7% | 0.49 |
| + widen rerank candidate pool 25 → 60 | **95.5%** | **94.7%** | **0.60** |

## **45.5% → 95.5%**

<img width="1214" height="688" alt="Filing Analyst evaluation results" src="https://github.com/user-attachments/assets/b9a97258-3f06-402e-b702-3a463b5bbd70" />

### Harder-set validation

The best configuration was re-evaluated on:

- **9 filings**
- **52 questions**
- additional industries and companies:
  - JPMorgan
  - Walmart
  - ExxonMobil
  - Coca-Cola

It scored **82.7% accuracy**.

The full ablation was **not** rerun on the 52-question set. On Groq's free tier, a single uncached naive-baseline pass required more than five hours of rate-limit backoff.

This larger run should therefore be read as **validation of the selected configuration**, not a second full ablation study.

---

## What changed — and why

### 1. Filing-aware retrieval

Scoping retrieval to the requested company and fiscal year improved:

- Accuracy: **45.5% → 54.5%**
- Recall@6: **26.3% → 42.1%**

10-Ks contain large amounts of semantically similar language. Without filing metadata, chunks from the wrong company or fiscal year can rank highly.

### 2. Cross-encoder reranking

Dense retrieval uses a bi-encoder: the question and chunks are embedded independently.

The reranker instead scores each `(question, chunk)` pair jointly using `ms-marco-MiniLM-L-6-v2`.

That improved:

- Accuracy: **54.5% → 63.6%**
- Recall@6: **42.1% → 57.9%**

### 3. Natural-language chunk headers

This was the largest semantic improvement.

Financial tables are structurally rich but often semantically sparse when embedded as raw text. A table such as:

```text
2026 | 2025 | 2024
32,681 | 27,441 | 26,974
```

does not embed especially well against a question like:

> What was NVIDIA's revenue in fiscal year 2026?

Before embedding, each chunk is enriched with metadata such as:

```text
NVIDIA CORP — fiscal year 2026 — Item 7 (table).
```

That improved:

- Accuracy: **63.6% → 77.3%**
- MRR: **0.23 → 0.49**

The result suggests that adding natural-language context to structurally sparse financial tables can materially improve dense retrieval.

### 4. Wider rerank candidate pool

Large multi-line financial-statement chunks often appeared around vector ranks **30–55**.

With a top-25 candidate pool, the cross-encoder never saw them. Expanding the pool from `25 → 60` allowed the reranker to recover those chunks.

Final initial-set result:

- Accuracy: **95.5%**
- Recall@6: **94.7%**
- MRR: **0.60**

### 5. The failed hybrid experiment stays

A hybrid experiment combining:

- dense vector retrieval
- PostgreSQL full-text search using `ts_rank_cd`
- reciprocal rank fusion (RRF)

reduced answer accuracy by roughly **18 percentage points**.

It remains available through:

```bash
--hybrid
```

and remains documented in the ablation history.

The likely failure mode is corpus-specific: PostgreSQL full-text ranking over these filings over-prioritized common financial terminology and boilerplate. Fusing that weak lexical ranking with the stronger dense retriever through RRF degraded the final ranking.

> **Failed experiments are evidence too. Removing them would make the ablation story misleading.**

**Note:** PostgreSQL `ts_rank_cd` is not BM25. This project refers to this experiment as PostgreSQL full-text search / lexical retrieval, not true BM25.

---

## Caveats

These numbers should not be treated as production-scale benchmark results.

### Small evaluation set

Current evaluation sizes:

- Initial ablation: **22 questions**
- Expanded validation: **52 questions**
- Target: **120+ questions**

### Judge calibration is not independent human validation

The automated judge was compared against Claude-applied labels:

- Cohen's κ = **1.0**
- n = **26**

That demonstrates self-consistency of the current evaluation pipeline. It does **not** demonstrate agreement with independent human annotators.

Independent human labeling is the highest-priority open evaluation task.

### Generator/judge dependence

Generation and judging currently use the same model family. Future work should separate generator, judge, and human ground truth more rigorously.

Per-question failures and the full experiment history live in [`evals/ablation.md`](evals/ablation.md).

---

## Architecture

```text
                         OFFLINE INGESTION

 SEC EDGAR
     │
     ▼
 Fetch 10-K
 (`edgar.py`)
     │
     ▼
 Structure-aware chunking
 (`chunker.py`)
     │
     ├── preserve tables
     ├── detect Item sections
     └── prepend semantic metadata
     │
     ▼
 Local embeddings
 (`all-MiniLM-L6-v2`)
     │
     ▼
 PostgreSQL + pgvector

                         QUESTION ANSWERING

 User question
      │
      ▼
 Dense vector retrieval
 top 60 candidates
      │
      ▼
 Cross-encoder reranking
      │
      ▼
 Top 6 chunks
      │
      ▼
 Grounded prompt
      │
      ▼
 LLM
      │
      ▼
 Answer + [n] citations

                         EVALUATION

 Golden question set
      │
      ▼
 answer(question)
      │
      ├──────────────► retrieval scoring
      │                Recall@K / MRR
      ▼
 Generated answer
      │
      ▼
 LLM judge
      │
      ▼
 Accuracy
      │
      ▼
 Versioned JSON result
```

---

## How it works

### Ingestion

A 10-K is one large HTML/iXBRL document.

`chunk_structured()` walks the document structure rather than applying a blind fixed-character window.

It:

- renders HTML tables into pipe-delimited grids
- keeps tables whole where possible
- tracks the current `Item N` section
- attaches company and fiscal-year metadata
- prepends a natural-language header before embedding

Chunks are embedded locally with `all-MiniLM-L6-v2` (384 dimensions, CPU) and stored in PostgreSQL using `pgvector`.

### Retrieval

`answer()` embeds the question and retrieves the 60 nearest chunks by cosine distance.

When filing scope is known, retrieval is restricted by `company + fiscal_year`.

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) then jointly scores each `(question, chunk)` pair. The top 6 chunks are passed to generation.

### Grounded generation

The generation prompt requires the model to answer only from retrieved excerpts.

When the required evidence is absent, the model must return:

```text
NOT_IN_FILING
```

Citation mode asks the model to emit `[n]` markers tied directly to retrieved excerpts.

### Caching and retries

All model calls go through `chat()`, which provides:

- deterministic request caching
- exponential backoff
- jitter
- `Retry-After` handling
- token logging
- latency logging

With temperature `0`, unchanged request/context pairs can be reused from cache, making repeated evaluation runs substantially cheaper.

### Observability

One trace row is recorded per `answer()` call.

Each trace ties together:

- question
- retrieved chunk IDs
- reranker scores
- final retrieval order
- generated answer
- refusal status
- `llm_call_id`
- token usage
- end-to-end latency

The API exposes `/traces` so failed questions can be inspected individually rather than reduced to a single aggregate accuracy number.

---

## Evaluation methodology

For each golden question:

```text
golden question
      │
      ▼
answer()
      │
      ├────────────► retrieval evaluation
      │              Recall@K / MRR
      ▼
generated answer
      │
      ▼
judge(answer, gold)
      │
      ▼
correct / incorrect
```

A valid refusal on an intentionally unanswerable question counts as correct.

Numerically equivalent answers with different formatting count as matches.

Retrieval relevance is evaluated independently from generation. A chunk is considered relevant when it contains every `must_contain` substring defined by the corresponding golden example.

---

## Repository layout

The orchestration layer is intentionally handwritten: approximately **1,250 non-blank lines across 18 files** under `sec10k/`.

No LangChain or LlamaIndex is used. The goal was to keep retrieval, reranking, caching, generation, tracing, and evaluation inspectable.

| File | Responsibility |
|---|---|
| `config.py` | typed settings from `.env` |
| `db.py` | PostgreSQL access, LLM call logging, traces |
| `cache.py` | SQLite deterministic response cache |
| `retry.py` | exponential backoff + jitter + `Retry-After` |
| `llm.py` | model calls, caching, retries, structured output |
| `embeddings.py` | local text embeddings |
| `search.py` | vector, PostgreSQL FTS, and hybrid retrieval |
| `rerank.py` | cross-encoder candidate reranking |
| `edgar.py` | SEC EDGAR filing retrieval |
| `chunker.py` | structure-aware HTML/table chunking |
| `ingest.py` | filing → chunks → embeddings → database |
| `answer.py` | retrieve → rerank → grounded generation |
| `goldens.py` | golden evaluation set loader |
| `judge.py` | automated answer judging |
| `evaluate.py` | accuracy, Recall@K, MRR |
| `calibration.py` | judge agreement / Cohen's κ |
| `app.py` | FastAPI demo |

Additional directories:

```text
scripts/   runnable entrypoints
tests/     mirrors sec10k/ responsibilities
evals/     goldens, ablations, evaluation JSON
web/       lightweight demo UI
```

---

## Quick start

### Prerequisites

- Python 3.12+
- Docker
- Groq API key
- SEC-compatible user agent

The project was developed primarily on WSL2 / Ubuntu. A Linux environment is recommended because the ML stack includes native dependencies such as PyTorch, tokenizers, and lxml.

### Install

```bash
python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
```

Configure:

```text
GROQ_API_KEY=your-key
SEC_USER_AGENT="your-name your@email.com"
```

SEC EDGAR requires an identifying user agent.

### Start PostgreSQL + pgvector

```bash
docker compose up -d
```

Default local database:

```text
host: localhost
port: 5433
database: sec10k
user: sec10k
password: sec10k
```

### Build the corpus

```bash
python scripts/build_corpus.py
```

This fetches 9 filings, performs structure-aware chunking, creates local embeddings, and writes roughly 3,300 rows to PostgreSQL.

No LLM calls are required during corpus construction.

---

## Web demo

```bash
uvicorn sec10k.app:app
```

Open `http://localhost:8000`.

The UI exposes:

- filing selection
- grounded answers
- clickable citations
- retrieved chunks
- reranker relevance scores
- filing metadata
- SEC Item section

Traces are available at `http://localhost:8000/traces`.

---

## Run evaluations

### Naive baseline

```bash
python scripts/run_eval.py "baseline" --sleep 18
```

### Best configuration

```bash
python scripts/run_eval.py \
  "scoped + rerank" \
  --scoped \
  --rerank \
  --sleep 18
```

`--sleep 18` helps stay within Groq free-tier rate limits.

Results are written to `evals/results/`. Cached requests normally complete in seconds on later runs.

---

## Calibrate the judge

```bash
python scripts/calibrate_judge.py
```

This compares the automated judge with labels in `evals/judge_calibration.yaml` and calculates agreement including Cohen's κ.

---

## Tests

Offline/non-live tests:

```bash
pytest -m "not live"
```

Full test suite:

```bash
pytest
```

The full suite also exercises live LLM and SEC EDGAR integrations.

---

## Stack

- Python 3.12+
- PostgreSQL 16
- `pgvector`
- `sentence-transformers`
- `all-MiniLM-L6-v2`
- `ms-marco-MiniLM-L-6-v2`
- Groq via OpenAI-compatible API
- FastAPI
- pytest
- Docker
- GitHub Actions

---

## Roadmap

### Evaluation

- [ ] Expand the golden set to **120+ questions**
- [ ] Add independent human annotation
- [ ] Measure human-vs-judge agreement
- [ ] Add confidence intervals
- [ ] Add bootstrap/significance analysis where appropriate
- [ ] Separate generator and judge model families

### Retrieval

- [ ] Add a **true BM25** baseline
- [ ] Re-evaluate lexical + dense fusion
- [ ] Compare alternative embedding models
- [ ] Compare alternative rerankers
- [ ] Break retrieval metrics down by question category
- [ ] Evaluate table and narrative retrieval separately

### Financial reasoning

- [ ] Add explicit numerical reasoning
- [ ] Add cross-section multi-hop questions
- [ ] Add cross-year comparisons
- [ ] Add cross-company comparisons

### Reproducibility / open source

- [ ] Publish versioned benchmark releases
- [ ] Add downloadable evaluation artifacts
- [ ] Add experiment configuration files
- [ ] Track model / latency / cost comparisons

---

## Research question

The project is ultimately exploring a broader question:

> **How much can careful retrieval design and measurement improve the reliability of LLM systems over structurally complex financial documents?**

The architecture is intentionally straightforward. The emphasis is on measurement, reproducibility, failure analysis, retrieval quality, grounding, and experimentation rather than framework complexity.

---

## Reproducibility

The complete ablation history is available in [`evals/ablation.md`](evals/ablation.md).

Individual evaluation runs are stored under `evals/results/`.

Failed experiments are retained alongside successful ones so the final configuration can be traced through the full development history.

---

## Citation

If you use Filing Analyst in research, benchmarking, or teaching, please cite the repository using the metadata in [`CITATION.cff`](CITATION.cff).

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
