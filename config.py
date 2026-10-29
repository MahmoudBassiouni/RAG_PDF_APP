"""
config.py
=========
Centralized configuration for the RAG PDF application.

Every tunable value used across the pipeline (models, chunking, retrieval,
paths, rate-limit behavior) lives here so it can be changed in one place
without touching pipeline logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into os.environ.
load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CACHE_DIR = DATA_DIR / "cache"

for _dir in (DOCUMENTS_DIR, VECTOR_DB_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Gemini API
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# NOTE ON MODEL NAMES (verify against https://ai.google.dev/gemini-api/docs/models
# before deploying, since Google renames/retires models over time):
#
#   - "gemini-3.1-flash-lite" is a STABLE, free-tier-eligible model (as of
#     mid-2026) with generous free-tier request limits. It is the default here.
#   - "gemini-2.5-flash" is also stable and free-tier-eligible today, but
#     Google has scheduled the whole Gemini 2.5 line for shutdown on
#     16 Oct 2026 — swap to it only if 3.x is unavailable in your region,
#     and plan to migrate off it before that date.
#
# Change this single value to switch which model the whole app uses.
GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite")

# Embedding model. gemini-embedding-001 is the current free-tier embedding
# model. Its free-tier quota is roughly 100 requests/min, 30k tokens/min,
# 1000 requests/day at the time of writing — verify current numbers at
# https://ai.google.dev/gemini-api/docs/rate-limits before relying on them.
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Dimensionality requested from the embedding model. gemini-embedding-001
# supports configurable output dimensionality; 768 keeps the vector store
# small while retaining strong retrieval quality for this use case.
EMBEDDING_OUTPUT_DIM = int(os.environ.get("EMBEDDING_OUTPUT_DIM", "768"))

# --------------------------------------------------------------------------
# Free-tier safety / retry behavior
# --------------------------------------------------------------------------
# Maximum retries on a transient error (429 / 503) before giving up.
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
# Base delay (seconds) for exponential backoff: delay = BASE_DELAY * 2**attempt
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "2.0"))
# Small pause between successive embedding batch calls to stay comfortably
# under free-tier requests-per-minute limits.
EMBEDDING_REQUEST_SLEEP_SECONDS = float(os.environ.get("EMBEDDING_REQUEST_SLEEP_SECONDS", "0.5"))
# How many chunks to embed per API call.
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "10"))

USER_FRIENDLY_QUOTA_MESSAGE = (
    "Gemini Free Tier quota/rate limit reached. Please wait a minute and try "
    "again, or reduce the number of requests (fewer/smaller PDFs, fewer "
    "questions in quick succession)."
)

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# Chunk size / overlap are measured in words (a simple, dependency-free,
# language-agnostic proxy for tokens). 900 words is roughly 1100-1300 tokens
# for English text, which sits inside the requested 800-1200 token range
# while leaving headroom. 150 words of overlap (~15%) keeps sentences that
# straddle a chunk boundary intact in at least one chunk, without bloating
# the number of chunks or the embedding bill.
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))
# Chunks with a cosine similarity below this are dropped from the context
# even if they are in the top-k, so we don't feed the LLM irrelevant text.
DEFAULT_RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.0"))

# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "pdf_rag_collection")

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
MIN_EXTRACTABLE_CHARS_PER_PAGE = 20  # below this, a page is treated as "empty"
MIN_EXTRACTABLE_RATIO_FOR_TEXT_PDF = 0.15  # fraction of pages that must have text


@dataclass
class RuntimeSettings:
    """Mutable, session-scoped overrides for values the GUI lets users tweak."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    extra: dict = field(default_factory=dict)
