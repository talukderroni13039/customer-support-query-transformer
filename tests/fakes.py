"""Fakes for every infrastructure Protocol. Deterministic, no network, no model downloads."""

import hashlib
import math
import re
import uuid
from typing import Any

import chromadb

from src.core.schemas.chunk import ChunkHit
from src.domains.support_search.schemas.chat_schemas import (
    AnswerContextArticle,
    AnswerKind,
    GeneratedAnswer,
)
from src.domains.support_search.schemas.rewrite_schemas import RewrittenQuery
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


class Spy:
    """Transparent proxy that records which methods were called (and their arguments)."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        if not callable(attribute):
            return attribute

        def recorder(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args))
            return attribute(*args, **kwargs)

        return recorder

    def called(self, name: str) -> int:
        return sum(1 for call_name, _ in self.calls if call_name == name)


class FakeReranker:
    """Scores by word overlap with the query (0-1) and returns the candidates re-sorted."""

    model_name = "fake-reranker"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ChunkHit]]] = []
        self.fail = False

    def rerank(self, query: str, candidates: list[ChunkHit]) -> list[ChunkHit]:
        self.calls.append((query, list(candidates)))
        if self.fail:
            raise RuntimeError("reranker crashed")
        query_words = set(words(query))
        rescored = [
            hit.with_score(len(query_words & set(words(hit.index_text))) / max(len(query_words), 1))
            for hit in candidates
        ]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)


class FakeRewriter:
    """Returns a scripted rewrite and records every call. Never touches the network."""

    def __init__(
        self,
        standalone_query: str = "rewritten query",
        needs_retrieval: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.standalone_query = standalone_query
        self.needs_retrieval = needs_retrieval
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery:
        self.calls.append((question, list(history)))
        if self.error is not None:
            raise self.error
        return RewrittenQuery(
            needs_retrieval=self.needs_retrieval,
            standalone_query=self.standalone_query if self.needs_retrieval else "",
        )


class FakeAnswerGenerator:
    """Returns a scripted answer and records every call. Never touches the network."""

    def __init__(
        self,
        kind: AnswerKind = AnswerKind.ANSWER,
        answer: str = "You can return it within 60 days.",
        source_article_ids: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.kind = kind
        self.answer_text = answer
        self.source_article_ids = source_article_ids
        self.error = error
        self.calls: list[tuple[str, list[str], list[AnswerContextArticle]]] = []

    async def answer(
        self, question: str, history: list[str], articles: list[AnswerContextArticle]
    ) -> GeneratedAnswer:
        self.calls.append((question, list(history), list(articles)))
        if self.error is not None:
            raise self.error
        needs_text = self.kind in (AnswerKind.ANSWER, AnswerKind.SMALL_TALK)
        sources = self.source_article_ids
        if sources is None:
            sources = [a.article_id for a in articles[:1]] if self.kind is AnswerKind.ANSWER else []
        return GeneratedAnswer(
            kind=self.kind,
            answer=self.answer_text if needs_text else "",
            source_article_ids=sources,
        )
