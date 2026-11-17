"""ChromaDB-backed vector store. Also the article source of truth (chunks + metadata)."""

import functools
from collections.abc import Callable
from typing import Any

from src.core.exceptions import VectorStoreError
from src.core.schemas.chunk import ChunkHit, StoredChunk
from src.domains.support_search.schemas.article_schemas import ArticleSummary

DEFAULT_COLLECTION = "support_kb"


def _wrap_errors[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(details={"operation": func.__name__}) from exc

    return wrapper


class ChromaStore:
    """Synchronous: callers must run it in a worker thread.

    `client` is any chromadb client (`PersistentClient` in the app, `EphemeralClient` in tests).
    Embeddings are always supplied by the caller, so no embedding function is attached.
    """

    def __init__(self, client: Any, collection_name: str = DEFAULT_COLLECTION) -> None:
        try:
            self._collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception as exc:
            raise VectorStoreError(details={"operation": "open_collection"}) from exc

    @classmethod
    def persistent(cls, path: str, collection_name: str = DEFAULT_COLLECTION) -> "ChromaStore":
        import chromadb

        try:
            client = chromadb.PersistentClient(path=path)
        except Exception as exc:
            raise VectorStoreError(details={"operation": "open_client"}) from exc
        return cls(client, collection_name)

    @_wrap_errors
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    @_wrap_errors
    def delete_article(self, article_id: str) -> int:
        ids = self._collection.get(where={"article_id": article_id}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    @_wrap_errors
    def get_article_chunks(self, article_id: str) -> list[StoredChunk]:
        result = self._collection.get(
            where={"article_id": article_id}, include=["documents", "metadatas"]
        )
        return self._to_stored(result)

    @_wrap_errors
    def get_all_chunks(self) -> list[StoredChunk]:
        return self._to_stored(self._collection.get(include=["documents", "metadatas"]))

    @_wrap_errors
    def list_articles(self, category: str | None = None) -> list[ArticleSummary]:
        where = {"category": category} if category else None
        metadatas = self._collection.get(where=where, include=["metadatas"])["metadatas"] or []
        grouped: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            entry = grouped.setdefault(str(meta["article_id"]), {"meta": meta, "count": 0})
            entry["count"] += 1
        return [
            ArticleSummary(
                id=article_id,
                title=str(entry["meta"]["title"]),
                category=str(entry["meta"].get("category") or "") or None,
                chunk_count=entry["count"],
                updated_at=str(entry["meta"]["updated_at"]),
            )
            for article_id, entry in sorted(grouped.items())
        ]

    @_wrap_errors
    def query(self, embedding: list[float], top_k: int) -> list[ChunkHit]:
        total = self._collection.count()
        if total == 0 or top_k < 1:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[ChunkHit] = []
        for chunk_id, document, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            hits.append(
                ChunkHit(
                    chunk_id=chunk_id,
                    article_id=str(meta["article_id"]),
                    title=str(meta["title"]),
                    category=str(meta.get("category") or "") or None,
                    heading=str(meta.get("heading") or ""),
                    text=document,
                    score=1.0 - float(distance),  # cosine distance → cosine similarity
                )
            )
        return hits

    @_wrap_errors
    def count(self) -> int:
        return int(self._collection.count())

    @staticmethod
    def _to_stored(result: dict[str, Any]) -> list[StoredChunk]:
        chunks = [
            StoredChunk(
                chunk_id=chunk_id,
                article_id=str(meta["article_id"]),
                title=str(meta["title"]),
                category=str(meta.get("category") or "") or None,
                heading=str(meta.get("heading") or ""),
                text=document,
                chunk_index=int(meta["chunk_index"]),
                created_at=str(meta["created_at"]),
                updated_at=str(meta["updated_at"]),
            )
            for chunk_id, document, meta in zip(
                result["ids"], result["documents"] or [], result["metadatas"] or [], strict=True
            )
        ]
        return sorted(chunks, key=lambda c: (c.article_id, c.chunk_index))
