"""What the structure-aware chunker must do. No network.

    pytest tests/test_chunker_structured.py -v
"""
from __future__ import annotations

from bs4 import BeautifulSoup

from sec10k.chunker import Block, chunk_structured, html_to_blocks, render_table

TABLE_HTML = """
<table>
  <tr><th></th><th>2025</th><th>2024</th><th>2023</th></tr>
  <tr><td>Net sales</td><td>416,161</td><td>391,035</td><td>383,285</td></tr>
  <tr><td>Net income</td><td>112,010</td><td>93,736</td><td>96,995</td></tr>
</table>
"""

DOC_HTML = f"""
<html><body>
<p>Item 1. Business</p>
<p>The Company designs and sells smartphones and services.</p>
<p>Item 8. Financial Statements</p>
<p>CONSOLIDATED STATEMENTS OF OPERATIONS</p>
{TABLE_HTML}
<p>See accompanying notes.</p>
</body></html>
"""


def test_render_table_keeps_all_columns_on_each_row():
    table = BeautifulSoup(TABLE_HTML, "lxml").find("table")
    out = render_table(table)
    lines = out.splitlines()
    assert "2025" in lines[0] and "2024" in lines[0] and "2023" in lines[0]
    # the data row has its label and all three figures together
    assert any("Net income" in ln and "93,736" in ln for ln in lines)


def test_html_to_blocks_makes_one_table_block():
    blocks = html_to_blocks(DOC_HTML)
    kinds = [b.kind for b in blocks]
    assert kinds.count("table") == 1
    table_block = next(b for b in blocks if b.kind == "table")
    # header and data survived in the same block
    assert "2023" in table_block.text
    assert "112,010" in table_block.text


def test_blocks_are_tagged_with_their_item_section():
    blocks = html_to_blocks(DOC_HTML)
    biz = next(b for b in blocks if "designs and sells" in b.text)
    assert biz.section == "Item 1"
    tbl = next(b for b in blocks if b.kind == "table")
    assert tbl.section == "Item 8"


def test_chunk_structured_keeps_table_whole_and_labeled():
    blocks = html_to_blocks(DOC_HTML)
    chunks = chunk_structured(blocks, target_chars=1500)

    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) == 1
    tc = table_chunks[0]
    assert tc.text.startswith("[Item 8]")
    # the whole table is in one chunk: header + both data rows
    assert "2025" in tc.text and "2024" in tc.text and "2023" in tc.text
    assert "416,161" in tc.text and "96,995" in tc.text


def test_prose_chunks_carry_a_section_prefix():
    blocks = html_to_blocks(DOC_HTML)
    chunks = chunk_structured(blocks, target_chars=1500)
    for c in chunks:
        assert c.text.splitlines()[0].startswith("[") and c.text.splitlines()[0].endswith("]")


def test_chunks_do_not_span_sections():
    blocks = [
        Block(kind="text", text="alpha " * 20, section="Item 1"),
        Block(kind="text", text="beta " * 20, section="Item 2"),
    ]
    chunks = chunk_structured(blocks, target_chars=100_000)  # huge target
    # even though both would fit, the section change forces a split
    assert len(chunks) == 2
    assert chunks[0].section == "Item 1"
    assert chunks[1].section == "Item 2"
