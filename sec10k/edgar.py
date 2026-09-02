"""Fetch 10-K filings from SEC EDGAR.

Three hops: ticker -> CIK -> filing list -> primary document. SEC requires a
descriptive User-Agent (see config.sec_user_agent) and rate-limits to ~10 req/s,
so _get() sleeps briefly before every call.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from pydantic import BaseModel

from sec10k.config import get_settings

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"


class FilingRef(BaseModel):
    """Everything needed to locate and label one filing."""

    cik: int
    company: str
    ticker: str
    form: str            # "10-K"
    accession: str       # dashed form, e.g. "0000320193-23-000106"
    filing_date: str     # ISO date the filing was submitted
    report_date: str     # ISO date of the fiscal period end
    primary_doc: str     # main document filename, e.g. "aapl-20230930.htm"
    doc_url: str          # full URL to primary_doc


def _get(url: str) -> bytes:
    """GET `url` with the SEC-required User-Agent header; return the raw body.
    Sleeps 0.15s first to stay well under SEC's 10 req/s limit."""
    time.sleep(0.15)
    req = urllib.request.Request(
        url, headers={"User-Agent": get_settings().sec_user_agent}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def cik_for_ticker(ticker: str) -> int:
    """Map a stock ticker to its CIK via SEC's company_tickers.json.
    Raises LookupError if the ticker isn't listed."""
    table = json.loads(_get(_TICKERS_URL))
    wanted = ticker.upper()
    for entry in table.values():
        if entry["ticker"] == wanted:
            return int(entry["cik_str"])
    raise LookupError(ticker)


def latest_10k(cik: int, *, ticker: str = "", before: str | None = None) -> FilingRef:
    """Build a FilingRef for the most recent 10-K filed by `cik` (optionally the
    most recent filed before the ISO date `before`). SEC's submissions JSON stores
    filings as parallel arrays, newest first. Raises LookupError if there's none.
    """
    data = json.loads(_get(_SUBMISSIONS_URL.format(cik=cik)))
    recent = data["filings"]["recent"]

    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        if before is not None and recent["filingDate"][i] >= before:
            continue

        accession = recent["accessionNumber"][i]
        primary_doc = recent["primaryDocument"][i]
        return FilingRef(
            cik=cik,
            company=data["name"],
            ticker=ticker.upper(),
            form=form,
            accession=accession,
            filing_date=recent["filingDate"][i],
            report_date=recent["reportDate"][i],
            primary_doc=primary_doc,
            doc_url=_ARCHIVE_URL.format(
                cik=cik,
                accession=accession.replace("-", ""),
                doc=primary_doc,
            ),
        )

    raise LookupError(f"no 10-K found for CIK {cik}")


def download_filing(ref: FilingRef, dest_dir: str | Path = "data/filings") -> Path:
    """Download ref.doc_url to {dest_dir}/{ref.accession}.html and return the Path.
    No-op if the file already exists (filings are immutable once accepted)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{ref.accession}.html"
    if not path.exists():
        path.write_bytes(_get(ref.doc_url))
    return path
