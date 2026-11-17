"""Retrieval stage: vector / BM25 / both fused with RRF, then rerank → ranked chunks."""

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from src.core.exceptions import RerankerUnavailableError
from src.core.schemas.chunk import ChunkHit
from src.domains.support_search.schemas.search_schemas import RetrievalMode, ScoreType
from src.embeddings.base import Embedder
from src.keyword_search.base import KeywordIndex
from src.reranking.base import Reranker
from src.utils.rank_fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES_PER_RETRIEVER = 20
DEFAULT_RERANK_CANDIDATES = 10


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
        reranker: Reranker | None = None,
        candidates_per_retriever: int = DEFAULT_CANDIDATES_PER_RETRIEVER,
        rrf_k: int = DEFAULT_RRF_K,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._candidates = candidates_per_retriever
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    @property
    def rerank_available(self) -> bool:
        return self._reranker is not None

    async def retrieve(
        self, query: str, mode: RetrievalMode, rerank: bool = False
    ) -> RetrievalOutcome:
        if rerank and self._reranker is None:
            raise RerankerUnavailableError()

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

        rerank_fallback = False
        rerank_ms: float | None = None
        if rerank and chunks:
            rerank_started = time.perf_counter()
            try:
                chunks = await self._rerank(query, chunks)
                score_type = ScoreType.RERANK
            except Exception as exc:
                # Resilience rule: a reranker failure never fails the search.
                rerank_fallback = True
                logger.warning(
                    "reranker failed, keeping prior order",
                    extra={"exception_type": type(exc).__name__},
                )
            rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 1)

        logger.info(
            "retrieval completed",
            extra={
                "retrieval": mode.value,
                "score_type": score_type.value,
                "vector_hits": len(vector_hits),
                "keyword_hits": len(keyword_hits),
                "ranked_chunks": len(chunks),
                "rerank": rerank,
                "rerank_fallback": rerank_fallback,
                "rerank_ms": rerank_ms,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return RetrievalOutcome(
            chunks=chunks, score_type=score_type, rerank_fallback=rerank_fallback
        )

    async def _rerank(self, query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        """Re-score only the top `rerank_candidates`; the result is those candidates re-sorted."""
        if self._reranker is None:  # pragma: no cover - guarded by `retrieve`
            raise RerankerUnavailableError()
        head = chunks[: self._rerank_candidates]
        by_id = {chunk.hit.chunk_id: chunk for chunk in head}
        reranked = await asyncio.to_thread(
            self._reranker.rerank, query, [chunk.hit for chunk in head]
        )
        if {hit.chunk_id for hit in reranked} != set(by_id) or len(reranked) != len(head):
            raise ValueError("reranker must return exactly the candidates it was given")
        return [replace(by_id[hit.chunk_id], hit=hit, rerank_score=hit.score) for hit in reranked]

    # --- synchronous helpers, always run in a worker thread ---------------------------------

    def _vector_search(self, query: str) -> list[ChunkHit]:
        embedding = self._embedder.embed_query(query)
        return self._vector_store.query(embedding, self._candidates)

    def _keyword_search(self, query: str) -> list[ChunkHit]:
        return self._keyword_index.search(query, self._candidates)
