"""Make one call through the real client and show that it was logged.

    python scripts/first_call.py
"""
from sec10k.db import get_connection
from sec10k.llm import chat


def main() -> None:
    result = chat(
        [
            {"role": "system", "content": "You answer questions about SEC filings concisely."},
            {"role": "user", "content": "In one sentence, what is Item 1A of a 10-K?"},
        ],
        tag="first-call",
    )

    print("answer :", result.text)
    print("tokens :", result.record.total_tokens,
          f"({result.record.prompt_tokens} in / {result.record.completion_tokens} out)")
    print("latency:", result.record.latency_ms, "ms")
    print("db row :", result.db_id)

    # Read it back straight from Postgres to prove it's really there.
    with get_connection() as conn:
        row = conn.execute(
            "SELECT created_at, provider, model, tag, total_tokens, latency_ms "
            "FROM llm_calls WHERE id = %s",
            (result.db_id,),
        ).fetchone()
    print("stored :", row)


if __name__ == "__main__":
    main()
