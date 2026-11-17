"""Retrieval stage: vector / BM25 / both fused with RRF → ranked chunks with stage debug."""

import asyncio
import logging
import time
from dataclasses import dataclass

from src.core.schemas.chunk import ChunkHit
from src.domains.support_search.schemas.search_schemas import RetrievalMode, ScoreType
from src.embeddings.base import Embedder
from src.keyword_search.base import KeywordIndex
from src.utils.rank_fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES_PER_RETRIEVER = 20


@dataclass(frozen=True)
class RankedChunk:
    """A chunk in final order. `hit.score` is the score of the stage named by `score_type`."""

    hit: ChunkHit
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalOutcome:
    chunks: list[RankedChunk]
    score_type: ScoreType
    rerank_fallback: bool = False


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        candidates_per_retriever: int = DEFAULT_CANDIDATES_PER_RETRIEVER,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._candidates = candidates_per_retriever
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, mode: RetrievalMode) -> RetrievalOutcome:
        started = time.perf_counter()
        vector_hits: list[ChunkHit] = []
        keyword_hits: list[ChunkHit] = []

        if mode is RetrievalMode.VECTOR:
            vector_hits = await asyncio.to_thread(self._vector_search, query)
            ranked, score_type = vector_hits, ScoreType.COSINE
        elif mode is RetrievalMode.KEYWORD:
            keyword_hits = await asyncio.to_thread(self._keyword_search, query)
            ranked, score_type = keyword_hits, ScoreType.BM25
        else:
            vector_hits, keyword_hits = await asyncio.gather(
                asyncio.to_thread(self._vector_search, query),
                asyncio.to_thread(self._keyword_search, query),
            )
            ranked = reciprocal_rank_fusion([vector_hits, keyword_hits], k=self._rrf_k)
            score_type = ScoreType.RRF

        vector_ranks = {hit.chunk_id: rank for rank, hit in enumerate(vector_hits, start=1)}
        keyword_ranks = {hit.chunk_id: rank for rank, hit in enumerate(keyword_hits, start=1)}
        chunks = [
            RankedChunk(
                hit=hit,
                vector_rank=vector_ranks.get(hit.chunk_id),
                keyword_rank=keyword_ranks.get(hit.chunk_id),
                rrf_score=hit.score if score_type is ScoreType.RRF else None,
            )
            for hit in ranked
        ]

        logger.info(
            "retrieval completed",
            extra={
                "retrieval": mode.value,
                "score_type": score_type.value,
                "vector_hits": len(vector_hits),
                "keyword_hits": len(keyword_hits),
                "ranked_chunks": len(chunks),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return RetrievalOutcome(chunks=chunks, score_type=score_type)

    # --- synchronous helpers, always run in a worker thread ---------------------------------

    def _vector_search(self, query: str) -> list[ChunkHit]:
        embedding = self._embedder.embed_query(query)
        return self._vector_store.query(embedding, self._candidates)

    def _keyword_search(self, query: str) -> list[ChunkHit]:
        return self._keyword_index.search(query, self._candidates)
