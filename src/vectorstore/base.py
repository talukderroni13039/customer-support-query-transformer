from typing import Any, Protocol

from src.core.schemas.chunk import ChunkHit, StoredChunk
from src.domains.support_search.schemas.article_schemas import ArticleSummary


class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def delete_article(self, article_id: str) -> int: ...

    def get_article_chunks(self, article_id: str) -> list[StoredChunk]: ...

    def get_all_chunks(self) -> list[StoredChunk]: ...  # feeds the BM25 rebuild

    def list_articles(self, category: str | None = None) -> list[ArticleSummary]: ...

    def query(self, embedding: list[float], top_k: int) -> list[ChunkHit]: ...

    def count(self) -> int: ...
