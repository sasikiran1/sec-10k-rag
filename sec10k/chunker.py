"""Turn a 10-K's HTML into retrievable chunks.

Structure-aware: tables are rendered whole (never split mid-grid), prose is packed
to a size target, and every chunk is tagged with the "Item N" section it's under.
(The naive fixed-window baseline this replaced is in git history, session 4.)
"""
from __future__ import annotations

import re
import warnings
from typing import Literal

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Tag
from pydantic import BaseModel

# Modern 10-Ks are inline-XBRL (XHTML). We parse them as HTML on purpose; the
# warning about that is just noise here.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Matches a 10-K section heading like "Item 1.", "Item 1A.", "Item 7A."
_ITEM_RE = re.compile(r"^\s*Item\s+(\d{1,2}[A-Z]?)\.?\s", re.IGNORECASE)


class Block(BaseModel):
    """One structural unit of the filing, in document order."""

    kind: Literal["text", "table"]
    text: str            # for a table, a grid-preserving rendering (see render_table)
    section: str = ""    # the most recent "Item N" heading this block falls under


class Chunk(BaseModel):
    """A retrieval unit produced from one or more Blocks."""

    section: str
    kind: Literal["prose", "table"]
    text: str            # includes a "[section]" prefix line


def render_table(table: Tag) -> str:
    """Render a <table> element to text that keeps rows and columns.

    - For each <tr>, collect cell text from its <th>/<td> children:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
    - Drop cells that are empty strings (10-K tables are full of spacer cells).
    - If any cells remain, join them with "  |  " -> one line per row.
    - Join the row-lines with "\n". Return "" if the table has no rows.
    """
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append("  |  ".join(cells))
    return "\n".join(rows)


def html_to_blocks(html: str) -> list[Block]:
    """Parse HTML into ordered Blocks, keeping tables intact.

    Each <table> is rendered by render_table() and fenced with @@TABLE/@@ENDTABLE
    sentinels so it survives get_text() as one unit; prose between sentinels is
    split on blank lines and section changes. `section` tracks the last "Item N".
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Swap every <table> for a sentinel string so it survives get_text() but stays
    # recognisable as an atomic unit.
    for table in soup.find_all("table"):
        rendered = render_table(table)
        table.replace_with(soup.new_string(f"\n@@TABLE\n{rendered}\n@@ENDTABLE\n"))

    blocks: list[Block] = []
    section = ""
    prose: list[str] = []
    table_buf: list[str] = []
    in_table = False

    def flush_prose() -> None:
        if prose:
            text = " ".join(prose).strip()
            if len(text) >= 2:
                blocks.append(Block(kind="text", text=text, section=section))
            prose.clear()

    for raw in soup.get_text("\n").splitlines():
        line = raw.strip()

        if not line:
            flush_prose()
            continue
        if line == "@@TABLE":
            flush_prose()
            in_table = True
            table_buf = []
            continue
        if line == "@@ENDTABLE":
            in_table = False
            blocks.append(Block(kind="table", text="\n".join(table_buf), section=section))
            table_buf = []
            continue
        if in_table:
            table_buf.append(line)
            continue

        m = _ITEM_RE.match(line)
        if m:
            flush_prose()
            section = f"Item {m.group(1).upper()}"
        prose.append(line)

    flush_prose()
    return blocks


def chunk_structured(blocks: list[Block], *, target_chars: int = 1500) -> list[Chunk]:
    """Pack Blocks into Chunks. Each table block is its own Chunk; text blocks
    accumulate up to `target_chars` then flush. A chunk never spans two sections,
    and every chunk's text starts with a "[<section>]" line."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_section: str | None = None

    def flush() -> None:
        if buf:
            label = buf_section or "document"
            body = "\n\n".join(buf)
            chunks.append(
                Chunk(section=buf_section or "", kind="prose", text=f"[{label}]\n{body}")
            )
            buf.clear()

    for b in blocks:
        if b.kind == "table":
            flush()
            label = b.section or "document"
            chunks.append(
                Chunk(section=b.section, kind="table", text=f"[{label}]\n{b.text}")
            )
            continue

        if buf and b.section != buf_section:
            flush()
        buf_section = b.section
        buf.append(b.text)
        if sum(len(x) for x in buf) >= target_chars:
            flush()

    flush()
    return chunks
