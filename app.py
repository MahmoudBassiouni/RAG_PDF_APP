"""
app.py
======
Streamlit GUI for the local PDF RAG application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    DOCUMENTS_DIR,
    RuntimeSettings,
)
from embeddings import MissingAPIKeyError
from pdf_processor import InvalidPDFError
from rag_pipeline import RAGPipeline, ScannedPDFDetected
from utils import GeminiAPIError, QuotaExceededError

st.set_page_config(page_title="Local PDF RAG (Gemini)", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Cached resources / session state
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_session_state() -> None:
    defaults = {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": DEFAULT_TOP_K,
        "relevance_threshold": DEFAULT_RELEVANCE_THRESHOLD,
        "last_query_result": None,
        "last_question": "",
        "pending_scanned_file": None,  # path awaiting user confirmation
        "status_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
pipeline = get_pipeline()


def current_settings() -> RuntimeSettings:
    return RuntimeSettings(
        chunk_size=st.session_state["chunk_size"],
        chunk_overlap=st.session_state["chunk_overlap"],
        top_k=st.session_state["top_k"],
        relevance_threshold=st.session_state["relevance_threshold"],
    )


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, MissingAPIKeyError):
        return str(exc)
    if isinstance(exc, QuotaExceededError):
        return str(exc)
    if isinstance(exc, GeminiAPIError):
        return f"There was a problem talking to the Gemini API: {exc}"
    if isinstance(exc, InvalidPDFError):
        return str(exc)
    return f"Something went wrong: {exc}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Local PDF RAG")
    st.caption("Gemini Free Tier · ChromaDB · Streamlit")

    st.subheader("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    allow_scanned = st.checkbox(
        "Index anyway if the PDF looks scanned (may produce poor results)",
        value=False,
        help="If a PDF has little extractable text, the app warns you instead "
        "of silently indexing an empty document. Check this to override.",
    )

    if st.button("📥 Upload & Index", use_container_width=True, disabled=uploaded_file is None):
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = DOCUMENTS_DIR / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())

        progress_bar = st.progress(0.0, text="Indexing...")

        def _progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Embedding chunks {done}/{total}...")

        try:
            result = pipeline.index_pdf(
                temp_path, current_settings(), allow_scanned=allow_scanned, progress_callback=_progress
            )
            progress_bar.progress(1.0, text="Done.")
            if result.warning:
                st.warning(result.warning)
            if result.was_cached:
                st.info(f"'{result.filename}' was already indexed and unchanged — reused existing embeddings.")
            else:
                st.success(
                    f"Indexed '{result.filename}': {result.num_pages} pages → {result.num_chunks} chunks."
                )
        except ScannedPDFDetected as exc:
            st.warning(str(exc))
            st.caption("Check 'Index anyway' above and click Upload & Index again if you still want to proceed.")
        except Exception as exc:  # noqa: BLE001
            st.error(friendly_error(exc))

    st.divider()
    st.subheader("2. Retrieval settings")
    st.session_state["top_k"] = st.number_input(
        "Number of retrieved chunks (Top-K)", min_value=1, max_value=20, value=st.session_state["top_k"]
    )
    st.session_state["chunk_size"] = st.number_input(
        "Chunk size (words)", min_value=100, max_value=3000, step=50, value=st.session_state["chunk_size"]
    )
    st.session_state["chunk_overlap"] = st.number_input(
        "Chunk overlap (words)", min_value=0, max_value=1000, step=25, value=st.session_state["chunk_overlap"]
    )
    st.caption("Chunk size/overlap changes apply to newly uploaded PDFs, not ones already indexed.")

    st.divider()
    st.subheader("3. Knowledge base")
    indexed_docs = pipeline.list_indexed_documents()
    if indexed_docs:
        for name, count in indexed_docs.items():
            st.write(f"• **{name}** — {count} chunks")
    else:
        st.caption("No documents indexed yet.")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
        pipeline.reset_knowledge_base()
        st.session_state["last_query_result"] = None
        st.success("Knowledge base cleared. Upload a new PDF to start over.")
        st.rerun()


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.header("Local PDF RAG Application")
st.caption(
    "PDF → Extraction → Cleaning → Chunking → Embeddings → ChromaDB → "
    "Retrieval → Gemini → Grounded Answer + Sources"
)

# Section 1 — Document overview
st.subheader("📚 Document(s)")
indexed_docs = pipeline.list_indexed_documents()
if indexed_docs:
    cols = st.columns(len(indexed_docs)) if len(indexed_docs) <= 4 else st.columns(4)
    for i, (name, count) in enumerate(indexed_docs.items()):
        with cols[i % len(cols)]:
            st.metric(label=name, value=f"{count} chunks")
else:
    st.info("No PDF indexed yet. Upload one from the sidebar to get started.")

st.divider()

# Section 2 — Ask questions
st.subheader("❓ Ask a question about your PDF")
question = st.text_input(
    "Ask a question about your PDF...", value=st.session_state["last_question"], key="question_input"
)
ask_clicked = st.button("Ask", type="primary", disabled=not question.strip())

if ask_clicked and question.strip():
    st.session_state["last_question"] = question
    with st.spinner("Retrieving relevant context and generating a grounded answer..."):
        try:
            result = pipeline.ask(question, current_settings())
            st.session_state["last_query_result"] = result
        except Exception as exc:  # noqa: BLE001
            st.session_state["last_query_result"] = None
            st.error(friendly_error(exc))

st.divider()

# Section 3 — Answer
st.subheader("💬 Answer")
result = st.session_state["last_query_result"]
if result is None:
    st.caption("Ask a question above to see a grounded answer here.")
else:
    st.write(result.answer)
    if result.sources:
        st.markdown("**Sources:**")
        for i, src in enumerate(result.sources, start=1):
            st.markdown(f"{i}. {src['source']} — Page {src['page']}")
    else:
        st.caption("No sources were retrieved for this answer.")

st.divider()

# Section 4 — Retrieval debugging
st.subheader("🔍 Retrieval debugging")
if result is None or not result.retrieved:
    st.caption("Retrieved chunks will appear here after you ask a question.")
else:
    for rank, chunk in enumerate(result.retrieved, start=1):
        with st.expander(
            f"Rank {rank} · {chunk.source} · Page {chunk.page} · similarity={chunk.similarity:.3f}"
        ):
            st.write(chunk.text)

st.divider()
st.caption(
    "Tip: if you see a Gemini Free Tier quota/rate limit message, wait a minute "
    "and try again, or reduce how many questions you ask in quick succession."
)
