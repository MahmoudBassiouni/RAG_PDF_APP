"""
embeddings.py
=============
Stage C of the RAG pipeline: turning chunk text (and questions) into
vector embeddings via the Gemini API, with batching, retry/backoff, and
free-tier-friendly pacing.

Uses the current unified `google-genai` SDK (`pip install google-genai`),
NOT the deprecated `google-generativeai` package.
"""

from __future__ import annotations

import time
from typing import Sequence

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_OUTPUT_DIM,
    EMBEDDING_REQUEST_SLEEP_SECONDS,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
)
from utils import GeminiAPIError, get_logger, with_retry

logger = get_logger(__name__)

_client: genai.Client | None = None


class MissingAPIKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is not set."""


def get_client() -> genai.Client:
    """Lazily create (and cache) the Gemini client, validating the API key first."""
    global _client
    if not GEMINI_API_KEY:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not set. Create a .env file (see .env.example) "
            "with your Gemini API key, or set the GEMINI_API_KEY environment variable."
        )
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _embed_batch(texts: Sequence[str], task_type: str) -> list[list[float]]:
    """Embed one batch of texts. Raises on API errors (retried by caller)."""
    client = get_client()
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_OUTPUT_DIM,
    )
    try:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=list(texts),
            config=config,
        )
    except (ClientError, ServerError) as exc:
        raise exc
    except Exception as exc:  # noqa: BLE001
        raise GeminiAPIError(f"Embedding request failed: {exc}") from exc

    return [list(embedding.values) for embedding in response.embeddings]


def embed_texts(
    texts: Sequence[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = EMBEDDING_BATCH_SIZE,
    progress_callback=None,
) -> list[list[float]]:
    """
    Embed a list of texts, batching requests and pacing them to stay under
    free-tier rate limits.

    task_type: "RETRIEVAL_DOCUMENT" for chunks going into the index,
               "RETRIEVAL_QUERY" for a user's question at search time.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = with_retry(_embed_batch, batch, task_type)
        all_embeddings.extend(batch_embeddings)
        if progress_callback is not None:
            progress_callback(min(start + batch_size, total), total)
        if start + batch_size < total:
            time.sleep(EMBEDDING_REQUEST_SLEEP_SECONDS)

    logger.info("Embedded %s text(s) using model '%s'.", total, GEMINI_EMBEDDING_MODEL)
    return all_embeddings


def embed_query(question: str) -> list[float]:
    """Embed a single user question for retrieval."""
    result = embed_texts([question], task_type="RETRIEVAL_QUERY", batch_size=1)
    return result[0]
