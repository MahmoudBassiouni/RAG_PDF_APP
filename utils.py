"""
utils.py
========
Small, shared utilities used across the pipeline: logging setup, file
hashing (for the "don't re-embed unchanged documents" cache), and a
generic retry/backoff decorator used for all Gemini API calls.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from typing import Any, Callable, TypeVar

from config import MAX_RETRIES, RETRY_BASE_DELAY_SECONDS, USER_FRIENDLY_QUOTA_MESSAGE

T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Never logs secrets (e.g. API keys)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("rag_app")


class QuotaExceededError(RuntimeError):
    """Raised when the Gemini free-tier quota/rate limit is hit repeatedly."""


class GeminiAPIError(RuntimeError):
    """Raised for non-quota Gemini API failures (invalid key, bad request, etc.)."""


def compute_file_hash(file_bytes: bytes) -> str:
    """Stable SHA-256 hash of file contents, used as a cache/index key."""
    return hashlib.sha256(file_bytes).hexdigest()


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of a 429 / quota-exceeded error from google-genai."""
    message = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (429,):
        return True
    keywords = ("429", "resource_exhausted", "quota", "rate limit", "rate_limit")
    return any(keyword in message for keyword in keywords)


def is_transient_server_error(exc: Exception) -> bool:
    """Best-effort detection of a transient 5xx error worth retrying."""
    message = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (500, 502, 503, 504):
        return True
    keywords = ("503", "502", "500", "unavailable", "internal error", "deadline exceeded")
    return any(keyword in message for keyword in keywords)


def with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY_SECONDS,
    **kwargs: Any,
) -> T:
    """
    Call `func(*args, **kwargs)` with exponential-backoff retries.

    - Retries on rate-limit (429) and transient server errors.
    - Raises QuotaExceededError with a friendly message once retries are
      exhausted on a rate-limit error.
    - Re-raises immediately (no retry) on anything else, e.g. invalid API
      key or a malformed request, since retrying won't help.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - we classify below
            last_exc = exc
            if is_rate_limit_error(exc):
                if attempt == max_retries:
                    logger.warning("Gemini rate limit exceeded after retries.")
                    raise QuotaExceededError(USER_FRIENDLY_QUOTA_MESSAGE) from exc
                delay = base_delay * (2**attempt)
                logger.info(
                    "Rate limited (attempt %s/%s). Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            if is_transient_server_error(exc):
                if attempt == max_retries:
                    raise GeminiAPIError(
                        f"Gemini API is temporarily unavailable: {exc}"
                    ) from exc
                delay = base_delay * (2**attempt)
                logger.info(
                    "Transient server error (attempt %s/%s). Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            # Non-retryable error (bad key, bad request, etc.)
            raise
    # Should be unreachable, but keeps type-checkers happy.
    assert last_exc is not None
    raise last_exc


def retry_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator form of `with_retry` for convenience."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return with_retry(func, *args, **kwargs)

    return wrapper
