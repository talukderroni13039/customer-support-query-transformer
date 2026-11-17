from typing import Protocol

from src.core.schemas.chunk import ChunkHit


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, candidates: list[ChunkHit]) -> list[ChunkHit]: ...
