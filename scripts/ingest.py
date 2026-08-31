"""Ingest one or more companies' latest 10-K filings.

    python scripts/ingest.py AAPL MSFT NVDA
"""
import sys

from sec10k.ingest import ingest_ticker


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL"]
    for ticker in tickers:
        r = ingest_ticker(ticker)
        state = "skipped (already present)" if r.skipped else f"{r.n_chunks} chunks"
        print(f"{ticker:6} {r.company:32} FY{r.fiscal_year}  {r.accession}  {state}")


if __name__ == "__main__":
    main()
