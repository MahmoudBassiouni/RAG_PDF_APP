"""
chunking.py
===========
Stage B of the RAG pipeline: splitting extracted page text into
overlapping, metadata-tagged chunks.

Design notes (why this approach, not naive character splitting):
- We split on paragraph boundaries first, then greedily pack paragraphs
  into a chunk until we approach `chunk_size` words. This keeps whole
  paragraphs together whenever they fit, instead of cutting mid-sentence.
- A paragraph longer than `chunk_size` on its own is split by sentence,
  and only falls back to a hard word-boundary split if a single sentence
  is still too long (rare, but guards against pathological input).
- Chunks never span a page boundary: a new page always starts a new
  chunk. This keeps the "source page" metadata for every chunk exact
  (no ambiguity about which page a chunk "really" came from), which the
  UI relies on for citations.
- `chunk_overlap` words from the end of one chunk are repeated at the
  start of the next chunk *within the same page*, so that text near a
  chunk boundary still has a good chance of being retrieved in full.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from pdf_processor import DocumentContent
from utils import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str
    metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        return {"source": self.source, "page": self.page, "chunk_id": self.chunk_id, **self.metadata}


def _split_into_paragraphs(page_text: str) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(page_text) if p.strip()]
    return paragraphs if paragraphs else ([page_text.strip()] if page_text.strip() else [])


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries, then on words as a last resort."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if sentence_len > max_words:
            # A single sentence is longer than the whole chunk budget: hard-split by words.
            words = sentence.split()
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
            continue
        if current_len + sentence_len > max_words and current:
            pieces.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += sentence_len
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack_paragraphs(
    paragraphs: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Greedily pack paragraphs (splitting over-long ones) into overlapping chunks."""
    # Normalize: break any paragraph that alone exceeds chunk_size.
    units: list[str] = []
    for para in paragraphs:
        if len(para.split()) > chunk_size:
            units.extend(_split_long_paragraph(para, chunk_size))
        else:
            units.append(para)

    chunks: list[str] = []
    current_units: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit.split())
        if current_len + unit_len > chunk_size and current_units:
            chunk_text = "\n\n".join(current_units)
            chunks.append(chunk_text)
            # Build overlap: take the trailing `chunk_overlap` words of this chunk
            # and seed the next chunk with them for continuity.
            overlap_words = chunk_text.split()[-chunk_overlap:] if chunk_overlap > 0 else []
            if overlap_words:
                current_units = [" ".join(overlap_words)]
                current_len = len(overlap_words)
            else:
                current_units, current_len = [], 0
        current_units.append(unit)
        current_len += unit_len

    if current_units:
        chunks.append("\n\n".join(current_units))

    return [c.strip() for c in chunks if c.strip()]


def chunk_document(
    document: DocumentContent,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Turn a DocumentContent (page-by-page text) into a flat list of Chunk
    objects, each tagged with source filename, page number, and a unique
    chunk_id.
    """
    if chunk_overlap >= chunk_size:
        logger.warning(
            "chunk_overlap (%s) >= chunk_size (%s); clamping overlap to chunk_size // 3.",
            chunk_overlap,
            chunk_size,
        )
        chunk_overlap = max(0, chunk_size // 3)

    all_chunks: list[Chunk] = []
    for page in document.pages:
        if not page.text.strip():
            continue  # empty page handled gracefully: simply contributes no chunks
        paragraphs = _split_into_paragraphs(page.text)
        if not paragraphs:
            continue
        page_chunks = _pack_paragraphs(paragraphs, chunk_size, chunk_overlap)
        for text in page_chunks:
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    source=document.filename,
                    page=page.page_number,
                    text=text,
                )
            )

    logger.info(
        "Chunked '%s' (%s pages) into %s chunks (chunk_size=%s, overlap=%s).",
        document.filename,
        document.num_pages,
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
