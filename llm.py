"""
llm.py
======
Stage E of the RAG pipeline: sending the retrieved context + question to
Gemini for grounded answer generation.
"""

from __future__ import annotations

from google.genai import types

from config import GEMINI_LLM_MODEL
from embeddings import get_client
from prompts import SYSTEM_PROMPT, build_user_prompt
from utils import GeminiAPIError, get_logger, with_retry

logger = get_logger(__name__)


def _generate(user_prompt: str) -> str:
    client = get_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,  # low temperature: favor grounded, literal answers
                max_output_tokens=1024,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - reraised for with_retry to classify
        raise exc

    text = getattr(response, "text", None)
    if not text:
        raise GeminiAPIError("Gemini returned an empty response.")
    return text


def generate_answer(question: str, retrieved_chunks) -> str:
    """
    Generate a grounded answer from the retrieved chunks. Raises
    QuotaExceededError (friendly message) or GeminiAPIError on failure —
    the GUI layer is responsible for turning these into friendly messages.
    """
    if not retrieved_chunks:
        return "I could not find sufficient information about this in the uploaded PDF."

    user_prompt = build_user_prompt(question, retrieved_chunks)
    answer = with_retry(_generate, user_prompt)
    logger.info("Generated answer for question: %r", question[:80])
    return answer
