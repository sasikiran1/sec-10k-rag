# sec-10k-rag

Question answering over SEC 10-K filings, with an evaluation harness that measures
whether each retrieval change actually helps.

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

## Tests

```
pytest -m "not live"     # no network; fast
pytest                   # includes live LLM + EDGAR calls
```
