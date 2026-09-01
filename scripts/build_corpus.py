"""Ingest the fixed evaluation corpus.

    python scripts/build_corpus.py

A small, varied set: three companies with different fiscal-year ends, plus one
prior-year Apple filing so the golden set can include cross-year comparisons.
Re-runnable — each filing is replaced, not duplicated.
"""
from sec10k.ingest import ingest_ticker

# (ticker, before)  -- `before` picks the newest 10-K filed before that date.
FILINGS = [
    ("AAPL", None),
    ("AAPL", "2024-06-01"),  # the prior fiscal year
    ("MSFT", None),
    ("NVDA", None),
]


def main() -> None:
    for ticker, before in FILINGS:
        r = ingest_ticker(ticker, before=before, replace=True)
        tag = "" if before is None else f" (before {before})"
        print(f"{ticker:6}{tag:20} {r.company:30} FY{r.fiscal_year}  "
              f"{r.accession}  {r.n_chunks} chunks")


if __name__ == "__main__":
    main()
