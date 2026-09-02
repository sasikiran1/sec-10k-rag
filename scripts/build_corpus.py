"""Ingest the fixed evaluation corpus.

    python scripts/build_corpus.py

A small, varied set: three companies with different fiscal-year ends, plus one
prior-year Apple filing so the golden set can include cross-year comparisons.
Re-runnable — each filing is replaced, not duplicated.
"""
from sec10k.ingest import ingest_ticker

# (ticker, before, cik)  -- `before` picks the newest 10-K filed before that date;
# `cik` overrides the ticker lookup. Mix of sectors (tech / bank / retail / energy
# / staples) so retrieval must cope with very different 10-K structures, plus a
# few prior years for cross-fiscal-year comparison questions.
FILINGS = [
    ("AAPL", None, None),
    ("AAPL", "2024-06-01", None),   # prior fiscal year
    ("MSFT", None, None),
    ("NVDA", None, None),
    ("NVDA", "2025-06-01", None),   # prior fiscal year
    ("JPM", None, None),            # bank
    ("WMT", None, None),            # retailer, Jan fiscal year end
    ("XOM", None, 34088),          # energy (ticker file points at a new holdco)
    ("KO", None, None),            # consumer staples
]


def main() -> None:
    for ticker, before, cik in FILINGS:
        r = ingest_ticker(ticker, before=before, cik=cik, replace=True)
        tag = "" if before is None else f" (before {before})"
        print(f"{ticker:6}{tag:20} {r.company:30} FY{r.fiscal_year}  "
              f"{r.accession}  {r.n_chunks} chunks")


if __name__ == "__main__":
    main()
