"""
vector_store.py
================
Stage D of the RAG pipeline: a local, persistent vector database.

Wraps ChromaDB (a lightweight, fully local, file-backed vector store —
no cloud account, no paid tier) behind a small interface tailored to
this app: add chunks with embeddings, similarity search, per-document
hash tracking (to skip re-embedding unchanged files), and a full reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from chunking import Chunk
from config import CHROMA_COLLECTION_NAME, VECTOR_DB_DIR
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int
    distance: float  # lower = more similar (cosine distance)

    @property
    def similarity(self) -> float:
        """Convert cosine distance in [0, 2] to a friendlier similarity in [0, 1]."""
        return max(0.0, 1.0 - (self.distance / 2.0))


class VectorStore:
    """Thin wrapper around a single persistent Chroma collection."""

    def __init__(self, persist_dir=VECTOR_DB_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- writes ------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata_dict() for c in chunks],
        )
        logger.info("Added %s chunks to vector store.", len(chunks))

    def delete_source(self, source_filename: str) -> None:
        """Remove all chunks belonging to a given source PDF filename."""
        self._collection.delete(where={"source": source_filename})
        logger.info("Deleted all chunks for source '%s'.", source_filename)

    def reset(self) -> None:
        """Wipe the entire knowledge base (used by the 'Clear Knowledge Base' button)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset.")

    # -- reads ---------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> dict[str, int]:
        """Return {filename: chunk_count} for every document currently indexed."""
        if self.count() == 0:
            return {}
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result["metadatas"]:
            source = meta.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def has_source(self, source_filename: str) -> bool:
        return source_filename in self.list_sources()

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        relevance_threshold: float = 0.0,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Semantic similarity search; returns the top-k chunks above the threshold."""
        if self.count() == 0:
            return []

        where = {"source": source_filter} if source_filter else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            item = RetrievedChunk(
                chunk_id=chunk_id,
                text=doc,
                source=meta.get("source", "unknown"),
                page=meta.get("page", -1),
                distance=dist,
            )
            if item.similarity >= relevance_threshold:
                retrieved.append(item)
        return retrieved
