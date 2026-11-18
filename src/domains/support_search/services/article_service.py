"""Article CRUD. ChromaDB is the source of truth; every write rebuilds the BM25 index."""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime

from src.core.exceptions import ArticleNotFoundError
from src.core.schemas.chunk import StoredChunk, compose_index_text
from src.domains.support_search.chunking.article_chunker import (
    DEFAULT_CHUNK_MAX_CHARS,
    chunk_article,
)
from src.domains.support_search.schemas.article_schemas import (
    ArticleCreate,
    ArticleDeleteResult,
    ArticleResult,
    ArticleSummary,
)
from src.embeddings.base import Embedder
from src.keyword_search.base import KeywordIndex
from src.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ArticleService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        write_lock: asyncio.Lock,
        chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._write_lock = write_lock  # serialises store write + BM25 rebuild across requests
        self._chunk_max_chars = chunk_max_chars

    async def create(self, payload: ArticleCreate) -> ArticleResult:
        async with self._write_lock:
            article_id = await asyncio.to_thread(self._new_id)
            return await asyncio.to_thread(self._write, article_id, payload, None)

    async def upsert(self, article_id: str, payload: ArticleCreate) -> ArticleResult:
        """Create or replace an article under a fixed ID (used by the seed script)."""
        async with self._write_lock:
            existing = await asyncio.to_thread(self._vector_store.get_article_chunks, article_id)
            created_at = existing[0].created_at if existing else None
            return await asyncio.to_thread(self._write, article_id, payload, created_at)

    async def update(self, article_id: str, payload: ArticleCreate) -> ArticleResult:
        async with self._write_lock:
            existing = await asyncio.to_thread(self._vector_store.get_article_chunks, article_id)
            if not existing:
                raise ArticleNotFoundError(details={"article_id": article_id})
            return await asyncio.to_thread(self._write, article_id, payload, existing[0].created_at)

    async def delete(self, article_id: str) -> ArticleDeleteResult:
        async with self._write_lock:
            removed = await asyncio.to_thread(self._delete_and_reindex, article_id)
        if removed == 0:
            raise ArticleNotFoundError(details={"article_id": article_id})
        logger.info("article deleted", extra={"article_id": article_id, "chunks": removed})
        return ArticleDeleteResult(id=article_id)

    async def get(self, article_id: str) -> ArticleResult:
        chunks = await asyncio.to_thread(self._vector_store.get_article_chunks, article_id)
        if not chunks:
            raise ArticleNotFoundError(details={"article_id": article_id})
        return self._to_result(chunks)

    async def list_articles(self, category: str | None = None) -> list[ArticleSummary]:
        return await asyncio.to_thread(self._vector_store.list_articles, category)

    # --- synchronous helpers, always run in a worker thread ---------------------------------

    def _new_id(self) -> str:
        while True:
            article_id = f"KB-{uuid.uuid4().hex[:6]}"
            if not self._vector_store.get_article_chunks(article_id):
                return article_id

    def _write(
        self, article_id: str, payload: ArticleCreate, created_at: str | None
    ) -> ArticleResult:
        started = time.perf_counter()
        now = _now()
        chunks = chunk_article(payload.content, self._chunk_max_chars)
        # Embed first: if the model fails, the stored article is left untouched.
        embeddings = self._embedder.embed_documents(
            [compose_index_text(payload.title, chunk.heading, chunk.text) for chunk in chunks]
        )
        stored = [
            StoredChunk(
                chunk_id=f"{article_id}::{chunk.index:04d}",
                article_id=article_id,
                title=payload.title,
                category=payload.category,
                heading=chunk.heading,
                text=chunk.text,
                chunk_index=chunk.index,
                created_at=created_at or now,
                updated_at=now,
            )
            for chunk in chunks
        ]
        self._vector_store.delete_article(article_id)
        self._vector_store.upsert(
            ids=[c.chunk_id for c in stored],
            embeddings=embeddings,
            documents=[c.text for c in stored],
            metadatas=[
                {
                    "article_id": c.article_id,
                    "title": c.title,
                    "category": c.category or "",
                    "heading": c.heading,
                    "chunk_index": c.chunk_index,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                }
                for c in stored
            ],
        )
        self._reindex()
        logger.info(
            "article indexed",
            extra={
                "article_id": article_id,
                "chunks": len(stored),
                "content_chars": len(payload.content),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return self._to_result(stored)

    def _delete_and_reindex(self, article_id: str) -> int:
        removed = self._vector_store.delete_article(article_id)
        if removed:
            self._reindex()
        return removed

    def _reindex(self) -> None:
        """Any article write must rebuild BM25 from the source of truth."""
        self._keyword_index.rebuild(self._vector_store.get_all_chunks())

    @staticmethod
    def _to_result(chunks: list[StoredChunk]) -> ArticleResult:
        ordered = sorted(chunks, key=lambda c: c.chunk_index)
        first = ordered[0]
        return ArticleResult(
            id=first.article_id,
            title=first.title,
            category=first.category,
            content="".join(c.text for c in ordered),
            chunk_count=len(ordered),
            created_at=first.created_at,
            updated_at=first.updated_at,
        )
