from typing import Protocol

from src.core.schemas.chunk import ChunkHit, StoredChunk


class KeywordIndex(Protocol):
    def rebuild(self, chunks: list[StoredChunk]) -> None: ...

    def search(self, query: str, top_k: int) -> list[ChunkHit]: ...

    def size(self) -> int: ...
