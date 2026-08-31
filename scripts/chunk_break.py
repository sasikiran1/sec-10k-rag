"""Run the naive chunker on a real 10-K and look at what it does to a table.

    python scripts/chunk_break.py [TICKER]   (default AAPL)
"""
import sys

from sec10k.chunker import chunk_fixed, html_to_text
from sec10k.edgar import cik_for_ticker, download_filing, latest_10k

SIZE = 1200
OVERLAP = 150
# A heading that sits directly above a flattened financial table.
NEEDLE = "CONSOLIDATED STATEMENTS OF OPERATIONS"
CONTEXT = 4  # how many chunks after the needle to print


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    ref = latest_10k(cik_for_ticker(ticker), ticker=ticker)
    path = download_filing(ref)
    print(f"{ref.company}  {ref.form}  filed {ref.filing_date}  ->  {path}")

    text = html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    print(f"stripped text: {len(text):,} chars")

    chunks = chunk_fixed(text, size=SIZE, overlap=OVERLAP)
    print(f"naive chunks : {len(chunks)} of ~{SIZE} chars\n")

    idx = next(
        (i for i, c in enumerate(chunks) if NEEDLE in c.upper()), None
    )
    if idx is None:
        print(f"'{NEEDLE}' not found — try another ticker")
        return

    for i in range(idx, min(idx + 1 + CONTEXT, len(chunks))):
        print("=" * 80)
        print(f"--- chunk {i} ---")
        print(chunks[i])

    print("\n" + "=" * 80)
    print("Look for: the column header ('2025 ... 2024 ... 2023') and the row")
    print("labels ('Net sales', 'Cost of sales', 'Operating income') landing in")
    print("different chunks from their numbers, and the table already flattened")
    print("into a run of digits with no rows or columns.")


if __name__ == "__main__":
    main()
