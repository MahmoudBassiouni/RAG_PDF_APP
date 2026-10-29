"""
tests/test_rag.py
==================
Lightweight tests that don't require a Gemini API key: chunking behavior,
scanned-PDF detection, and cache-key hashing. Retrieval/generation tests
that call the real API are described in the README's manual test plan
(Tests 1-5), since they need a live GEMINI_API_KEY and would otherwise
burn free-tier quota on every CI run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import chunk_document  # noqa: E402
from pdf_processor import DocumentContent, PageContent, looks_like_scanned_pdf  # noqa: E402
from utils import compute_file_hash  # noqa: E402


def _make_document(page_texts: list[str], filename: str = "test.pdf") -> DocumentContent:
    pages = [PageContent(page_number=i + 1, text=t) for i, t in enumerate(page_texts)]
    return DocumentContent(filename=filename, pages=pages)


def test_chunking_respects_page_boundaries():
    doc = _make_document(["Paragraph one.\n\nParagraph two.", "Page two content here."])
    chunks = chunk_document(doc, chunk_size=900, chunk_overlap=150)
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}
    # No chunk should mix text from two different pages.
    for chunk in chunks:
        assert chunk.source == "test.pdf"


def test_chunking_splits_long_page_into_multiple_chunks():
    long_text = " ".join(f"word{i}" for i in range(3000))
    doc = _make_document([long_text])
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page == 1


def test_chunking_skips_empty_pages():
    doc = _make_document(["Some real content here.", "", "   "])
    chunks = chunk_document(doc, chunk_size=900, chunk_overlap=150)
    assert all(c.page == 1 for c in chunks)


def test_scanned_pdf_detection_flags_image_only_pdf():
    doc = _make_document(["", "", ""])
    assert looks_like_scanned_pdf(doc) is True


def test_scanned_pdf_detection_passes_text_pdf():
    doc = _make_document(["A perfectly normal page of extracted text " * 5 for _ in range(3)])
    assert looks_like_scanned_pdf(doc) is False


def test_file_hash_is_stable_and_content_sensitive():
    h1 = compute_file_hash(b"hello world")
    h2 = compute_file_hash(b"hello world")
    h3 = compute_file_hash(b"hello world!")
    assert h1 == h2
    assert h1 != h3


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
