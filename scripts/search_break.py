"""Show where naive vector search falls down.

    python scripts/search_break.py

Seeds ~12 chunks, then runs queries whose correct answer we know, and prints what
retrieval actually returns. Cleans up after itself.
"""
from sec10k.db import get_connection
from sec10k.search import add_chunks, search

SOURCE = "break-demo"

CHUNKS = [
    # -- the two negation-sensitive pairs --
    "We did not repurchase any shares of common stock during fiscal 2023.",
    "The company repurchased $76.6 billion of its common stock during fiscal 2023.",
    # -- question-echo vs the actual answer --
    "Item 7 includes a discussion of the principal factors affecting our gross margin.",
    "Gross margin expanded to 44.1% from 43.3%, driven by a mix shift toward Services.",
    # -- distractors --
    "Total net sales decreased 3% to $383.3 billion in fiscal 2023.",
    "Services set an all-time revenue record of $85.2 billion, up 9% year over year.",
    "The effective tax rate for fiscal 2023 was 14.7%.",
    "Cash, cash equivalents and marketable securities totaled $162.1 billion.",
    "The Board declared a cash dividend of $0.24 per share of common stock.",
    "Greater China net sales were $72.6 billion, roughly flat versus the prior year.",
    "Research and development expense was $29.9 billion, up 14% year over year.",
    "The company employs approximately 161,000 full-time equivalent employees.",
]

QUERIES = [
    ("Did the company refrain from buying back its own stock in 2023?",
     "should surface the 'did NOT repurchase' chunk, not the $76.6B repurchase chunk"),
    ("What were the factors affecting gross margin?",
     "the answer is the 44.1% chunk; the 'Item 7 includes a discussion' chunk just echoes the question"),
    ("How did Greater China revenue change from fiscal 2022 to fiscal 2023?",
     "needs a fiscal-2022 figure that isn't in the corpus at all — watch it answer anyway"),
]


def main() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chunks WHERE source = %s", (SOURCE,))
    add_chunks([(SOURCE, i, t) for i, t in enumerate(CHUNKS)])

    try:
        for query, note in QUERIES:
            print("\n" + "=" * 80)
            print("Q:", query)
            print("(", note, ")")
            for rank, hit in enumerate(search(query, k=3), start=1):
                print(f"  {rank}. score={hit.score:.3f}  {hit.text}")
    finally:
        with get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE source = %s", (SOURCE,))


if __name__ == "__main__":
    main()
