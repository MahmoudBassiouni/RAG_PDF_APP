"""
pdf_processor.py
================
Stage A of the RAG pipeline: document ingestion.

Responsible for validating a PDF, extracting text page-by-page, cleaning
that text, and detecting scanned/image-only PDFs that yield little or no
extractable text (so the app can warn the user instead of silently
indexing an empty document).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import MIN_EXTRACTABLE_CHARS_PER_PAGE, MIN_EXTRACTABLE_RATIO_FOR_TEXT_PDF
from utils import get_logger

logger = get_logger(__name__)


class InvalidPDFError(ValueError):
    """Raised when the uploaded file is not a readable PDF."""


class ScannedPDFWarning(Warning):
    """Raised (as an exception used for control flow) for scanned/image-only PDFs."""


@dataclass
class PageContent:
    page_number: int  # 1-indexed, matches what a human would call "page N"
    text: str


@dataclass
class DocumentContent:
    filename: str
    pages: list[PageContent]

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace and strip common PDF extraction noise."""
    if not raw_text:
        return ""
    # Collapse repeated whitespace/newlines, drop stray form-feed/control chars.
    text = raw_text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def validate_pdf(file_path: Path) -> None:
    """Raise InvalidPDFError if the file cannot be opened as a PDF."""
    if file_path.suffix.lower() != ".pdf":
        raise InvalidPDFError(f"Unsupported file type: '{file_path.suffix}'. Only .pdf is supported.")
    try:
        reader = PdfReader(str(file_path))
        _ = len(reader.pages)  # forces parsing of the page tree
    except (PdfReadError, Exception) as exc:  # noqa: BLE001
        raise InvalidPDFError(f"Could not read '{file_path.name}' as a PDF: {exc}") from exc


def extract_pdf_content(file_path: Path) -> DocumentContent:
    """
    Extract text from every page of the PDF, preserving page numbers and
    handling empty pages gracefully (they become empty-string entries,
    not errors).
    """
    validate_pdf(file_path)
    reader = PdfReader(str(file_path))

    pages: list[PageContent] = []
    for index, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from page %s of %s: %s", index + 1, file_path.name, exc)
            raw_text = ""
        pages.append(PageContent(page_number=index + 1, text=_clean_text(raw_text)))

    return DocumentContent(filename=file_path.name, pages=pages)


def looks_like_scanned_pdf(document: DocumentContent) -> bool:
    """
    Heuristic: a PDF is "scanned/image-only" if too few pages have any
    meaningful extractable text.
    """
    if document.num_pages == 0:
        return True
    pages_with_text = sum(
        1 for p in document.pages if len(p.text) >= MIN_EXTRACTABLE_CHARS_PER_PAGE
    )
    ratio = pages_with_text / document.num_pages
    return ratio < MIN_EXTRACTABLE_RATIO_FOR_TEXT_PDF
