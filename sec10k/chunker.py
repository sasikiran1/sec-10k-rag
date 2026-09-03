"""Convert SEC 10-K HTML into structure-aware retrieval chunks.

Tables remain intact, prose is packed to a target size, and each chunk retains
its SEC Item section.
"""
from __future__ import annotations

import re
import warnings
from typing import Literal

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Tag
from pydantic import BaseModel

# Modern 10-Ks are inline-XBRL/XHTML and are intentionally parsed as HTML here.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_ITEM_RE = re.compile(r"^\s*Item\s+(\d{1,2}[A-Z]?)\.?\s", re.IGNORECASE)


class Block(BaseModel):
    """A structural unit of a filing in document order."""

    kind: Literal["text", "table"]
    text: str
    section: str = ""


class Chunk(BaseModel):
    """A retrieval unit produced from one or more blocks."""

    section: str
    kind: Literal["prose", "table"]
    text: str


def render_table(table: Tag) -> str:
    """Render an HTML table as a row-preserving pipe-delimited text grid."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append("  |  ".join(cells))
    return "\n".join(rows)


def html_to_blocks(html: str) -> list[Block]:
    """Parse filing HTML into ordered prose and table blocks."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Sentinels preserve table boundaries while the surrounding HTML is flattened.
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
    """Pack blocks into section-bounded prose chunks and atomic table chunks."""
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
