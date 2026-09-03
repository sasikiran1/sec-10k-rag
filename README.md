# Filing Analyst

### Evaluation-driven RAG for SEC 10-K filings

> A structure-aware retrieval and evaluation system for answering financial questions over SEC 10-K filings — built to measure whether each retrieval change actually improves answer quality.

https://github.com/user-attachments/assets/741f2984-3ba0-4afb-b172-4467cbd7e1ae

---

## Why this project exists

Most RAG demos stop once the system produces plausible answers.

This project starts there.

**Filing Analyst** treats retrieval as an empirical system:

```text
hypothesis
   ↓
implementation
   ↓
evaluation
   ↓
failure analysis
   ↓
ablation
   ↓
re-evaluation
```

Every retrieval improvement is implemented separately and evaluated against a hand-verified question set.

Failed experiments stay in the history.

The goal is not:

> "The answers look good."

The goal is:

> "Naive retrieval answered 45.5% of the evaluation set correctly. Here is exactly which change improved or degraded that number."

---

## Results

### Initial ablation

22 hand-verified questions across four SEC 10-K filings:

* Apple FY2023
* Apple FY2025
* Microsoft FY2026
* NVIDIA FY2026

Generation and judging use Groq `openai/gpt-oss-120b` at temperature `0`.

Retrieval metrics (`Recall@6`, `MRR`) are calculated across the 19 non-refusal questions.

**Each row corresponds to a separate experiment and commit.**

| Configuration                              |  Accuracy |  Recall@6 |    MRR   |
| ------------------------------------------ | :-------: | :-------: | :------: |
| Naive vector retrieval, unscoped           |   45.5%   |   26.3%   |   0.11   |
| + scope retrieval by company + fiscal year |   54.5%   |   42.1%   |   0.18   |
| + cross-encoder reranking                  |   63.6%   |   57.9%   |   0.23   |
| + natural-language chunk headers           |   77.3%   |   73.7%   |   0.49   |
| + rerank candidate pool 25 → 60            | **95.5%** | **94.7%** | **0.60** |

### **45.5% → 95.5%**

<img width="1214" height="688" alt="Evaluation results" src="https://github.com/user-attachments/assets/b9a97258-3f06-402e-b702-3a463b5bbd70" />

---

## Harder-set validation

The best configuration was then evaluated on a larger and more diverse set:

* **9 SEC filings**
* **52 questions**
* additional companies and industries:

  * JPMorgan
  * Walmart
  * ExxonMobil
  * Coca-Cola

The same configuration achieved:

# **82.7% accuracy**

The full ablation was not repeated over all 52 questions because a single uncached baseline run on Groq's free tier required more than five hours of rate-limit backoff.

The larger evaluation therefore serves as **out-of-sample validation of the selected configuration**, not a second complete ablation.

---

## Key findings

### 1. Scoping retrieval helped immediately

Restricting retrieval to the requested company and fiscal year increased:

```text
Accuracy
45.5% → 54.5%

Recall@6
26.3% → 42.1%
```

Financial filings contain large amounts of semantically similar language.

Without metadata scoping, chunks from unrelated filings can rank highly despite belonging to the wrong company or fiscal period.

---

### 2. Cross-encoder reranking improved retrieval quality

Initial vector retrieval uses a bi-encoder:

```text
question → embedding
chunk    → embedding
```

Those embeddings are compared independently.

The cross-encoder instead evaluates:

```text
(question, chunk)
```

together.

Adding `ms-marco-MiniLM-L-6-v2` increased accuracy:

```text
54.5% → 63.6%
```

and Recall@6:

```text
42.1% → 57.9%
```

---

### 3. Natural-language chunk headers produced the largest semantic improvement

Financial tables often contain very little natural-language context.

For example:

```text
2026 | 2025 | 2024
32,681 | 27,441 | 26,974
```

does not embed particularly well against:

> "What was NVIDIA's revenue in fiscal year 2026?"

Each chunk is therefore enriched before embedding with metadata such as:

```text
NVIDIA CORP — fiscal year 2026 — Item 7 (table).
```

This improved:

```text
Accuracy
63.6% → 77.3%

MRR
0.23 → 0.49
```

The result suggests that **adding semantic context to structurally sparse financial tables can materially improve dense retrieval**.

---

### 4. Increasing the rerank candidate pool recovered buried financial tables

Large multi-line financial-statement chunks often appeared around vector ranks:

```text
30–55
```

With a candidate pool of only 25, the cross-encoder never saw them.

Increasing the pool:

```text
25 → 60
```

allowed the reranker to recover those chunks.

Final initial-set performance:

```text
Accuracy: 95.5%
Recall@6: 94.7%
MRR:      0.60
```

---

### 5. Hybrid retrieval made the system worse

A hybrid experiment combining:

* dense vector retrieval
* PostgreSQL full-text retrieval using `ts_rank_cd`
* reciprocal rank fusion

was also evaluated.

It **reduced accuracy by approximately 18 percentage points**.

The experiment remains available through:

```bash
--hybrid
```

and remains documented in the ablation history.

The likely explanation is that PostgreSQL full-text ranking over this corpus over-weighted common filing terminology and boilerplate such as:

```text
Apple
revenue
company
year
```

Fusing a weak lexical ranker with a stronger dense retriever through RRF degraded the resulting ranking.

The failed experiment is intentionally retained.

> Deleting a failed experiment is how ablation tables become misleading.

---

## Important caveats

The results should not be interpreted as production-scale benchmark numbers.

### Dataset size

The current evaluation sets contain:

```text
Initial: 22 questions
Expanded: 52 questions
Target: 120+ questions
```

The sample is still small.

---

### Judge validation

The LLM judge is currently calibrated against Claude-applied labels:

```text
Cohen's κ = 1.0
n = 26
```

This demonstrates **self-consistency of the automated evaluation pipeline**, not agreement with independent human annotators.

Independent human labeling is the highest-priority evaluation improvement.

---

### Model dependence

Generation and automated judging currently use the same model family through Groq.

Future work should separate:

```text
generator
judge
human ground truth
```

more rigorously.

---

## System architecture

```text
                         OFFLINE INGESTION

 SEC EDGAR
     │
     ▼
 Fetch 10-K
 edgar.py
     │
     ▼
 Structure-aware parsing
 chunker.py
     │
     ├── preserve tables
     ├── detect Item sections
     └── prepend semantic metadata
     │
     ▼
 Local embeddings
 all-MiniLM-L6-v2
     │
     ▼
 PostgreSQL + pgvector


                         QUESTION ANSWERING

 User question
      │
      ▼
 Dense retrieval
 top 60 candidates
      │
      ▼
 Cross-encoder reranker
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
      ▼
 Generated answer
      │
      ├──────────────► retrieval scoring
      │                Recall@K / MRR
      ▼
 LLM judge
      │
      ▼
 Accuracy
      │
      ▼
 JSON experiment result
```

---

## How ingestion works

SEC 10-K filings are delivered as large HTML/iXBRL documents.

Blind fixed-character chunking can break important relationships such as:

```text
year header
    ↓
financial metric
    ↓
value
```

especially inside tables.

`chunk_structured()` therefore walks the filing structure instead.

It:

* detects HTML tables
* converts tables into pipe-delimited grids
* keeps tables intact where possible
* identifies the current `Item N` section
* associates chunks with company and fiscal year
* prepends natural-language metadata before embedding

Example metadata:

```text
NVIDIA CORP — fiscal year 2026 — Item 7 (table).
```

Chunks are embedded locally using:

```text
all-MiniLM-L6-v2
384 dimensions
CPU inference
```

and stored in a `pgvector` column.

---

## Retrieval pipeline

The default retrieval path is:

```text
question
   │
   ▼
vector search
top 60
   │
   ▼
cross-encoder reranking
   │
   ▼
top 6
   │
   ▼
generation
```

Vector retrieval uses cosine similarity.

The reranker uses:

```text
ms-marco-MiniLM-L-6-v2
```

to jointly score each:

```text
(question, candidate chunk)
```

pair.

Optional PostgreSQL full-text + RRF hybrid retrieval remains available for reproducing the failed experiment.

---

## Grounded generation

The generation prompt instructs the model to answer **only from retrieved filing excerpts**.

If the retrieved context does not contain enough evidence, the model must return:

```text
NOT_IN_FILING
```

The web interface enables citation mode.

Answers contain markers such as:

```text
[1]
[2]
```

which link directly to the retrieved chunks used to generate the response.

---

## Caching and rate-limit handling

All model calls go through:

```python
chat()
```

which provides:

* deterministic request caching
* exponential-backoff retries
* jitter
* `Retry-After` handling
* latency logging
* token logging

Because generation uses:

```text
temperature = 0
```

an unchanged request can be safely reused from the local cache.

This makes repeated evaluation runs dramatically cheaper.

---

## Observability

Every `answer()` request creates a trace record containing:

* user question
* retrieved chunk IDs
* reranker scores
* final retrieval order
* generated answer
* refusal status
* model call ID
* token usage
* end-to-end latency

The API exposes:

```text
/traces
```

for inspecting previous requests.

This makes individual failures debuggable instead of reducing evaluation to a single aggregate accuracy number.

---

## Evaluation methodology

For each golden question:

```text
golden question
      │
      ▼
answer()
      │
      ▼
generated answer
      │
      ├─────────► retrieval evaluation
      │            Recall@K
      │            MRR
      ▼
judge(answer, gold)
      │
      ▼
correct / incorrect
```

A valid refusal on an intentionally unanswerable question counts as correct.

Numeric answers with equivalent values but different formatting are considered equivalent.

Retrieval relevance is evaluated separately from answer generation.

A chunk is considered relevant when it contains all required `must_contain` evidence specified by the golden example.

This keeps retrieval evaluation relatively stable even when chunk boundaries change.

---

## Repository structure

The orchestration layer is intentionally handwritten.

Approximately **1,250 non-blank lines across 18 files** live under `sec10k/`.

No LangChain or LlamaIndex is used.

The purpose was to make every retrieval, ranking, caching and evaluation step inspectable.

| File             | Responsibility                                   |
| ---------------- | ------------------------------------------------ |
| `config.py`      | Typed application settings                       |
| `db.py`          | PostgreSQL access, LLM-call logging, traces      |
| `cache.py`       | SQLite deterministic response cache              |
| `retry.py`       | Retry/backoff behavior                           |
| `llm.py`         | Model calls, caching, retries, structured output |
| `embeddings.py`  | Local embedding generation                       |
| `search.py`      | Vector, PostgreSQL FTS and hybrid retrieval      |
| `rerank.py`      | Cross-encoder candidate reranking                |
| `edgar.py`       | SEC EDGAR filing retrieval                       |
| `chunker.py`     | Structure-aware HTML and table chunking          |
| `ingest.py`      | Filing → chunks → embeddings → database          |
| `answer.py`      | End-to-end RAG pipeline                          |
| `goldens.py`     | Golden evaluation dataset loading                |
| `judge.py`       | Automated answer judging                         |
| `evaluate.py`    | Accuracy, Recall@K and MRR evaluation            |
| `calibration.py` | Judge agreement / Cohen's κ                      |
| `app.py`         | FastAPI API and demo application                 |

Additional directories:

```text
scripts/
    runnable entrypoints

tests/
    mirrors sec10k/ responsibilities

evals/
    goldens
    ablation history
    evaluation JSON

web/
    lightweight demo frontend
```

---

## Quick start

### Prerequisites

* Python 3.12+
* Docker
* Groq API key
* SEC-compatible user agent

The project was developed primarily on WSL2 / Ubuntu.

A Linux environment is recommended because parts of the ML stack depend on native libraries such as:

* PyTorch
* tokenizers
* lxml

---

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

---

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

---

### Build the corpus

```bash
python scripts/build_corpus.py
```

This:

```text
fetches 9 filings
      ↓
structure-aware parsing
      ↓
chunking
      ↓
local embeddings
      ↓
~3,300 PostgreSQL rows
```

No LLM calls are required during corpus construction.

---

## Run the web demo

```bash
uvicorn sec10k.app:app
```

Open:

```text
http://localhost:8000
```

Select a filing or search across all filings.

The interface displays:

* generated answer
* clickable citations
* retrieved chunks
* cross-encoder relevance scores
* filing metadata
* SEC Item section

Request traces are available at:

```text
http://localhost:8000/traces
```

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

`--sleep 18` helps remain within Groq free-tier rate limits.

Results are written to:

```text
evals/results/
```

with one JSON artifact per run.

Cached requests normally complete in seconds on subsequent runs.

---

## Calibrate the automated judge

```bash
python scripts/calibrate_judge.py
```

This compares automated judge output against the labels stored in:

```text
evals/judge_calibration.yaml
```

and calculates agreement including Cohen's κ.

---

## Tests

Run offline/non-live tests:

```bash
pytest -m "not live"
```

Requirements:

```text
PostgreSQL running
No external network calls
```

Run the complete suite:

```bash
pytest
```

The full suite also exercises live LLM and SEC EDGAR integrations.

---

## Stack

```text
Python 3.12+
PostgreSQL 16
pgvector
sentence-transformers
Groq / OpenAI-compatible API
FastAPI
pytest
Docker
GitHub Actions
```

Embedding model:

```text
all-MiniLM-L6-v2
```

Reranker:

```text
ms-marco-MiniLM-L-6-v2
```

---

## Current limitations

* Evaluation dataset remains relatively small.
* Automated judge has not yet been independently validated by human annotators.
* Full ablations have not been repeated across the expanded 52-question dataset.
* Retrieval currently focuses on individual filings rather than cross-document analytical reasoning.
* Numerical reasoning is delegated primarily to the generation model.
* PostgreSQL FTS is not a true BM25 implementation.
* Current experiments use a limited set of generator/judge models.

---

## Research and engineering roadmap

### Evaluation

* [ ] Expand evaluation set to **120+ questions**
* [ ] Add independent human annotation
* [ ] Measure human-vs-judge agreement
* [ ] Add confidence intervals
* [ ] Add bootstrapped significance testing where appropriate
* [ ] Separate generator and judge model families

### Retrieval

* [ ] Compare dense retrieval against a **true BM25 implementation**
* [ ] Re-evaluate dense + lexical fusion
* [ ] Evaluate alternative embedding models
* [ ] Evaluate alternative rerankers
* [ ] Analyze retrieval sensitivity by question category

### Financial-document reasoning

* [ ] Add explicit numerical reasoning
* [ ] Add multi-hop questions across sections
* [ ] Add cross-year comparisons
* [ ] Add cross-company comparisons
* [ ] Evaluate table-specific retrieval separately from narrative retrieval

### Reproducibility

* [ ] Publish versioned benchmark releases
* [ ] Add `CITATION.cff`
* [ ] Add reproducible experiment configuration files
* [ ] Add downloadable evaluation artifacts
* [ ] Publish model/cost/latency comparisons

---

## What this project is testing

Filing Analyst is ultimately an experiment in a broader question:

> **How much can careful retrieval design and measurement improve the reliability of LLM systems over structurally complex financial documents?**

The RAG architecture itself is intentionally straightforward.

The emphasis is on:

```text
measurement
reproducibility
failure analysis
retrieval quality
grounding
experimentation
```

rather than framework complexity.

---

## Reproducibility

The complete evaluation history is available in:

[`evals/ablation.md`](evals/ablation.md)

Individual experiment outputs are stored under:

```text
evals/results/
```

Failed experiments are retained alongside successful ones so that the reported final configuration can be traced back through the full development history.

---

## License

Add an explicit open-source license before encouraging external reuse.

For a permissive open-source project, MIT or Apache-2.0 are common options.

Choose the license deliberately based on how you want others to use the project.
