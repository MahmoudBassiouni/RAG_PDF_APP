"""
rag_pipeline.py
================
Orchestrates the full RAG pipeline:

    PDF -> extraction -> cleaning -> chunking -> embeddings -> vector store
    (indexing)

    question -> query embedding -> similarity search -> top-k context
    -> Gemini -> grounded answer + sources (querying)

Also owns the simple document-hash cache that avoids re-extracting /
re-embedding a PDF that is already indexed and unchanged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from chunking import Chunk, chunk_document
from config import CACHE_DIR, DOCUMENTS_DIR, RuntimeSettings
from embeddings import embed_query, embed_texts
from llm import generate_answer
from pdf_processor import DocumentContent, extract_pdf_content, looks_like_scanned_pdf
from utils import compute_file_hash, get_logger
from vector_store import RetrievedChunk, VectorStore

logger = get_logger(__name__)

_CACHE_INDEX_PATH = CACHE_DIR / "indexed_documents.json"


@dataclass
class IndexResult:
    filename: str
    num_pages: int
    num_chunks: int
    was_cached: bool
    warning: str | None = None


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)  # for the "Sources:" list
    retrieved: list[RetrievedChunk] = field(default_factory=list)  # for debug panel


class ScannedPDFDetected(Exception):
    """Raised when a PDF has too little extractable text to index safely."""


def _load_cache_index() -> dict:
    if _CACHE_INDEX_PATH.exists():
        try:
            return json.loads(_CACHE_INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Cache index file was corrupted; starting fresh.")
    return {}


def _save_cache_index(index: dict) -> None:
    _CACHE_INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


class RAGPipeline:
    """Stateful pipeline: holds a VectorStore and the document-hash cache."""

    def __init__(self) -> None:
        self.store = VectorStore()
        self._cache = _load_cache_index()

    # -- indexing ------------------------------------------------------------

    def index_pdf(
        self,
        file_path: Path,
        settings: RuntimeSettings,
        allow_scanned: bool = False,
        progress_callback=None,
    ) -> IndexResult:
        """
        Ingest one PDF: extract -> chunk -> embed -> store.
        Skips re-embedding if this exact file (by content hash) is already indexed.
        """
        file_bytes = file_path.read_bytes()
        file_hash = compute_file_hash(file_bytes)
        filename = file_path.name

        cached_entry = self._cache.get(filename)
        if cached_entry and cached_entry.get("hash") == file_hash and self.store.has_source(filename):
            logger.info("'%s' is unchanged and already indexed; skipping re-embedding.", filename)
            return IndexResult(
                filename=filename,
                num_pages=cached_entry.get("num_pages", 0),
                num_chunks=cached_entry.get("num_chunks", 0),
                was_cached=True,
            )

        document: DocumentContent = extract_pdf_content(file_path)

        warning = None
        if looks_like_scanned_pdf(document):
            warning = (
                "This PDF appears to be scanned/image-based and contains little "
                "extractable text. OCR may be required for good results."
            )
            if not allow_scanned:
                raise ScannedPDFDetected(warning)

        # If this document was previously indexed under an older version, clear it first.
        if self.store.has_source(filename):
            self.store.delete_source(filename)

        chunks: list[Chunk] = chunk_document(
            document, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        if not chunks:
            self._cache[filename] = {
                "hash": file_hash,
                "num_pages": document.num_pages,
                "num_chunks": 0,
            }
            _save_cache_index(self._cache)
            return IndexResult(
                filename=filename,
                num_pages=document.num_pages,
                num_chunks=0,
                was_cached=False,
                warning=warning or "No text could be extracted from this PDF.",
            )

        texts = [c.text for c in chunks]
        vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", progress_callback=progress_callback)
        self.store.add_chunks(chunks, vectors)

        self._cache[filename] = {
            "hash": file_hash,
            "num_pages": document.num_pages,
            "num_chunks": len(chunks),
        }
        _save_cache_index(self._cache)

        # Persist a copy of the source PDF for reference/debugging.
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = DOCUMENTS_DIR / filename
        if dest.resolve() != file_path.resolve():
            shutil.copyfile(file_path, dest)

        return IndexResult(
            filename=filename,
            num_pages=document.num_pages,
            num_chunks=len(chunks),
            was_cached=False,
            warning=warning,
        )

    # -- querying --------------------------------------------------------------

    def ask(self, question: str, settings: RuntimeSettings) -> QueryResult:
        """Embed the question, retrieve top-k chunks, and generate a grounded answer."""
        if self.store.count() == 0:
            return QueryResult(
                answer="Please upload and index a PDF before asking a question.",
                sources=[],
                retrieved=[],
            )

        query_vector = embed_query(question)
        retrieved = self.store.query(
            query_vector,
            top_k=settings.top_k,
            relevance_threshold=settings.relevance_threshold,
        )

        answer = generate_answer(question, retrieved)

        # De-duplicated, ordered list of (source, page) for the "Sources:" display.
        seen = set()
        sources = []
        for item in retrieved:
            key = (item.source, item.page)
            if key not in seen:
                seen.add(key)
                sources.append({"source": item.source, "page": item.page})

        return QueryResult(answer=answer, sources=sources, retrieved=retrieved)

    # -- housekeeping ------------------------------------------------------------

    def reset_knowledge_base(self) -> None:
        """Clear the vector store, the cache index, and any copied source PDFs."""
        self.store.reset()
        self._cache = {}
        _save_cache_index(self._cache)
        for f in DOCUMENTS_DIR.glob("*.pdf"):
            f.unlink(missing_ok=True)
        logger.info("Knowledge base fully reset.")

    def list_indexed_documents(self) -> dict[str, int]:
        return self.store.list_sources()
