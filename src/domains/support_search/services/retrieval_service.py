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
# Off by default: on the benchmark a 0.01 cut-off removed correct articles (full-pipeline hit@3
# 0.97 → 0.90), because bge-reranker-base scores correct and off-topic chunks in the same range.
DEFAULT_RERANK_MIN_SCORE = 0.0
# Upper bound when the candidate window is widened to reach `min_articles` distinct articles.
MAX_CANDIDATES_PER_RETRIEVER = 320
# Each retriever's top hits are always shown to the reranker (see select_rerank_pool).
RERANK_POOL_RETRIEVER_TOP = 3


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


def distinct_articles(chunks: list[RankedChunk]) -> int:
    return len({chunk.hit.article_id for chunk in chunks})


def select_rerank_pool(
    chunks: list[RankedChunk],
    pool_size: int,
    min_articles: int,
    retriever_top: int = RERANK_POOL_RETRIEVER_TOP,
) -> list[RankedChunk]:
    """Pick at most `pool_size` chunks to rerank, keeping the ranked order.

    Normally this is simply the top `pool_size` chunks. Two things can change it:

    - A chunk that one retriever ranked in its top `retriever_top` is always included. Rank
      fusion favours chunks that both retrievers found, so the best vector hit can land just
      outside the pool when keyword search missed it (or the other way round); the reranker is
      the judge, so it must at least see each retriever's best candidates.
    - When the pool covers fewer than `min_articles` articles (one long article can fill the
      whole window), duplicates are swapped for the best chunk of the next unseen articles.

    The pool never grows beyond `pool_size`.
    """
    head = chunks[:pool_size]
    target = min(min_articles, pool_size, distinct_articles(chunks))

    def is_retriever_top(chunk: RankedChunk) -> bool:
        return any(
            rank is not None and rank <= retriever_top
            for rank in (chunk.vector_rank, chunk.keyword_rank)
        )

    head_ids = {chunk.hit.chunk_id for chunk in head}
    must = [c for c in chunks if is_retriever_top(c)][:pool_size]
    if distinct_articles(head) >= target and all(c.hit.chunk_id in head_ids for c in must):
        return head

    chosen: dict[str, RankedChunk] = {}
    seen_articles: set[str] = set()
    for chunk in must:
        slots_left = pool_size - len(chosen)
        if slots_left <= 0:
            break
        still_needed = max(target - len(seen_articles), 0)
        # A retriever-top chunk is taken unless it would crowd out an article still needed.
        if chunk.hit.article_id not in seen_articles or slots_left - 1 >= still_needed:
            chosen[chunk.hit.chunk_id] = chunk
            seen_articles.add(chunk.hit.article_id)
    for chunk in chunks:  # then the best chunk of further articles, up to the target
        if len(seen_articles) >= target or len(chosen) == pool_size:
            break
        if chunk.hit.article_id not in seen_articles:
            seen_articles.add(chunk.hit.article_id)
            chosen[chunk.hit.chunk_id] = chunk
    for chunk in chunks:  # then fill the remaining slots in ranked order
        if len(chosen) == pool_size:
            break
        chosen.setdefault(chunk.hit.chunk_id, chunk)
    return [chunk for chunk in chunks if chunk.hit.chunk_id in chosen]


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
        rerank_min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._candidates = candidates_per_retriever
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._rerank_min_score = rerank_min_score

    @property
    def rerank_available(self) -> bool:
        return self._reranker is not None

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        rerank: bool = False,
        min_articles: int = 1,
    ) -> RetrievalOutcome:
        """Ranked chunks for `query`.

        `min_articles` is how many distinct articles the caller wants to show. Results are
        grouped by article afterwards, so a fixed chunk window can hold too few articles; the
        window is widened (doubling) until it holds enough or the retrievers are exhausted.
        """
        if rerank and self._reranker is None:
            raise RerankerUnavailableError()

        started = time.perf_counter()
        candidates = self._candidates
        embedding: list[float] | None = None
        if mode is not RetrievalMode.KEYWORD:
            embedding = await asyncio.to_thread(self._embedder.embed_query, query)

        while True:
            chunks, score_type, sizes = await self._ranked_chunks(
                query, embedding, mode, candidates
            )
            exhausted = all(size < candidates for size in sizes)
            if (
                distinct_articles(chunks) >= min_articles
                or exhausted
                or candidates >= MAX_CANDIDATES_PER_RETRIEVER
            ):
                break
            candidates = min(candidates * 2, MAX_CANDIDATES_PER_RETRIEVER)

        rerank_fallback = False
        rerank_ms: float | None = None
        below_min_score = 0
        if rerank and chunks:
            rerank_started = time.perf_counter()
            try:
                reranked = await self._rerank(query, chunks, min_articles)
                kept = [c for c in reranked if (c.rerank_score or 0.0) >= self._rerank_min_score]
                below_min_score = len(reranked) - len(kept)
                chunks, score_type = kept, ScoreType.RERANK
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
                "candidates_per_retriever": candidates,
                "retriever_hits": sizes,
                "ranked_chunks": len(chunks),
                "rerank": rerank,
                "rerank_fallback": rerank_fallback,
                "rerank_below_min_score": below_min_score,
                "rerank_ms": rerank_ms,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return RetrievalOutcome(
            chunks=chunks, score_type=score_type, rerank_fallback=rerank_fallback
        )

    async def _ranked_chunks(
        self, query: str, embedding: list[float] | None, mode: RetrievalMode, candidates: int
    ) -> tuple[list[RankedChunk], ScoreType, list[int]]:
        vector_hits: list[ChunkHit] = []
        keyword_hits: list[ChunkHit] = []
        if mode is RetrievalMode.VECTOR:
            assert embedding is not None  # noqa: S101 - set by `retrieve` for vector modes
            vector_hits = await asyncio.to_thread(self._vector_store.query, embedding, candidates)
            ranked, score_type, sizes = vector_hits, ScoreType.COSINE, [len(vector_hits)]
        elif mode is RetrievalMode.KEYWORD:
            keyword_hits = await asyncio.to_thread(self._keyword_index.search, query, candidates)
            ranked, score_type, sizes = keyword_hits, ScoreType.BM25, [len(keyword_hits)]
        else:
            assert embedding is not None  # noqa: S101 - set by `retrieve` for vector modes
            vector_hits, keyword_hits = await asyncio.gather(
                asyncio.to_thread(self._vector_store.query, embedding, candidates),
                asyncio.to_thread(self._keyword_index.search, query, candidates),
            )
            ranked = reciprocal_rank_fusion([vector_hits, keyword_hits], k=self._rrf_k)
            score_type, sizes = ScoreType.RRF, [len(vector_hits), len(keyword_hits)]

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
        return chunks, score_type, sizes

    async def _rerank(
        self, query: str, chunks: list[RankedChunk], min_articles: int
    ) -> list[RankedChunk]:
        """Re-score at most `rerank_candidates` chunks; the result is those chunks re-sorted."""
        if self._reranker is None:  # pragma: no cover - guarded by `retrieve`
            raise RerankerUnavailableError()
        pool = select_rerank_pool(chunks, self._rerank_candidates, min_articles)
        by_id = {chunk.hit.chunk_id: chunk for chunk in pool}
        reranked = await asyncio.to_thread(
            self._reranker.rerank, query, [chunk.hit for chunk in pool]
        )
        if {hit.chunk_id for hit in reranked} != set(by_id) or len(reranked) != len(pool):
            raise ValueError("reranker must return exactly the candidates it was given")
        return [replace(by_id[hit.chunk_id], hit=hit, rerank_score=hit.score) for hit in reranked]
