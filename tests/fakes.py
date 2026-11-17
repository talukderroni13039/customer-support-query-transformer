"""Fakes for every infrastructure Protocol. Deterministic, no network, no model downloads."""

import hashlib
import math
import re
import uuid
from typing import Any

import chromadb

from src.vectorstore.chroma_store import ChromaStore

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words get a high cosine similarity."""

    model_name = "fake-embedder"

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in words(text):
            digest = hashlib.md5(word.encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]


def make_chroma_store() -> ChromaStore:
    """In-memory Chroma. Ephemeral clients share one process-wide store, so use a unique name."""
    return ChromaStore(chromadb.EphemeralClient(), f"test-{uuid.uuid4().hex}")


def chunk_metadata(
    article_id: str,
    index: int,
    *,
    title: str = "Title",
    category: str = "",
    heading: str = "",
    stamp: str = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "article_id": article_id,
        "title": title,
        "category": category,
        "heading": heading,
        "chunk_index": index,
        "created_at": stamp,
        "updated_at": stamp,
    }
