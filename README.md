# sec-10k-rag

Question answering over SEC 10-K filings, with an evaluation harness that measures
whether each retrieval change actually helps.

## Dev setup

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill in GROQ_API_KEY
docker compose up -d
```

Postgres listens on `localhost:5433` (db/user/pass all `sec10k`).
