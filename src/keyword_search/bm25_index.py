"""In-memory BM25 keyword index (rank_bm25 `BM25Okapi`), rebuilt from the vector store."""

import math
import threading
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from src.core.schemas.chunk import ChunkHit, StoredChunk
from src.keyword_search.tokenizer import tokenize


class _PositiveIdfBM25Okapi(BM25Okapi):  # type: ignore[misc]
    """BM25Okapi with the Lucene-style IDF `ln(1 + (N - n + 0.5) / (n + 0.5))`.

    The stock rank_bm25 IDF is zero or negative for a term found in half or more of the
    documents, which makes true matches score <= 0 on a small knowledge base.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


@dataclass(frozen=True)
class _IndexState:
    """One immutable snapshot. Searches read a snapshot; rebuilds replace it."""

    chunks: tuple[StoredChunk, ...] = ()
    bm25: Any | None = None


class BM25Index:
    """Synchronous and CPU-bound: callers must run it in a worker thread.

    `rebuild` builds a complete new index and then swaps one reference, so a concurrent
    `search` sees either the old index or the new one, never a half-built one. A lock
    serialises concurrent rebuilds.
    """

    def __init__(self) -> None:
        self._state = _IndexState()
        self._rebuild_lock = threading.Lock()

    def rebuild(self, chunks: list[StoredChunk]) -> None:
        with self._rebuild_lock:
            corpus = [tokenize(chunk.index_text) for chunk in chunks]
            bm25 = _PositiveIdfBM25Okapi(corpus) if any(corpus) else None
            self._state = _IndexState(chunks=tuple(chunks), bm25=bm25)  # atomic swap

    def search(self, query: str, top_k: int) -> list[ChunkHit]:
        state = self._state  # single read: this search uses one consistent snapshot
        tokens = tokenize(query)
        if state.bm25 is None or not tokens or top_k < 1:
            return []
        scores = state.bm25.get_scores(tokens)
        ranked = sorted(
            ((float(score), i) for i, score in enumerate(scores) if score > 0),
            key=lambda pair: (-pair[0], pair[1]),
        )
        hits: list[ChunkHit] = []
        for score, i in ranked[:top_k]:
            chunk = state.chunks[i]
            hits.append(
                ChunkHit(
                    chunk_id=chunk.chunk_id,
                    article_id=chunk.article_id,
                    title=chunk.title,
                    category=chunk.category,
                    heading=chunk.heading,
                    text=chunk.text,
                    score=score,
                )
            )
        return hits

    def size(self) -> int:
        return len(self._state.chunks)
