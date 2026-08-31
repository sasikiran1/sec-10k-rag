"""Turn a 10-K's HTML into retrievable chunks.

Two chunkers live here. `chunk_fixed` is the naive baseline — a fixed-size sliding
window over stripped text. It exists to be measured against and to show, on a real
filing, how blind splitting shreds tables. The structure-aware chunker comes next.
"""
from __future__ import annotations

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Modern 10-Ks are inline-XBRL (XHTML). We parse them as HTML on purpose; the
# warning about that is just noise here.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def html_to_text(html: str) -> str:
    """Strip HTML tags to plain text.

    Deliberately crude for now: this flattens tables into runs of numbers with no
    column headers attached — which is part of what we want to see fail.

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def chunk_fixed(text: str, *, size: int = 1200, overlap: int = 150) -> list[str]:
    """Naive chunker: slide a `size`-character window with `overlap` between windows.

    - Start at 0. Each chunk is text[start : start + size].
    - Next start = start + size - overlap  (so consecutive chunks share `overlap` chars).
    - Stop once start >= len(text).
    - Drop a trailing chunk that is only whitespace.
    - For text shorter than `size`, return [text] (one chunk).

    No awareness of sentences, tables, or headings — that's the point.
    """
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        piece = text[start : start + size]
        if piece.strip():
            chunks.append(piece)
        start += step
    return chunks
