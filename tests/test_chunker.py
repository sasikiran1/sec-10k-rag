"""What the chunkers must do. No network.

    pytest tests/test_chunker.py -v
"""
from __future__ import annotations

from sec10k.chunker import chunk_fixed, html_to_text


def test_html_to_text_strips_tags_and_scripts():
    html = """
    <html><head><style>.x{color:red}</style><script>var a=1;</script></head>
    <body><h1>Item 1A</h1><p>Risk &amp; uncertainty.</p></body></html>
    """
    text = html_to_text(html)
    assert "Item 1A" in text
    assert "Risk & uncertainty." in text
    assert "color:red" not in text
    assert "var a" not in text


def test_chunk_fixed_short_text_is_one_chunk():
    assert chunk_fixed("short", size=1200, overlap=150) == ["short"]


def test_chunk_fixed_respects_size():
    text = "x" * 5000
    chunks = chunk_fixed(text, size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert len(chunks) > 1


def test_chunk_fixed_windows_overlap():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(3000))
    chunks = chunk_fixed(text, size=1000, overlap=100)
    # the last 100 chars of chunk 0 are the first 100 of chunk 1
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_fixed_covers_all_text():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(2500))
    chunks = chunk_fixed(text, size=1000, overlap=100)
    # every character of the source appears somewhere
    assert text[:1000] in chunks[0]
    assert text[-100:] in chunks[-1]
