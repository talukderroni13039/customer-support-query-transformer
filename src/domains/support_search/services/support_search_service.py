"""Orchestrates the pipeline: query → retrieval → group chunks by article."""

import logging
import time

from src.core.exceptions import KnowledgeBaseEmptyError, RerankerUnavailableError
from src.core.schemas.chunk import chunk_body
from src.domains.support_search.schemas.search_schemas import (
    RetrievalMode,
    RetrievedArticle,
    ScoreType,
    SearchRequest,
    StageDebug,
    SupportSearchResult,
)
from src.domains.support_search.services.query_service import QueryService
from src.domains.support_search.services.retrieval_service import RankedChunk, RetrievalService
from src.keyword_search.base import KeywordIndex

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
SNIPPET_MAX_CHARS = 300

_BASE_SCORE_TYPE = {
    RetrievalMode.VECTOR: ScoreType.COSINE,
    RetrievalMode.KEYWORD: ScoreType.BM25,
    RetrievalMode.HYBRID: ScoreType.RRF,
}


def make_snippet(chunk: RankedChunk) -> str:
    """Chunk body without its heading line, cut to a readable length."""
    hit = chunk.hit
    body = " ".join(chunk_body(hit.heading, hit.text).split())
    if len(body) <= SNIPPET_MAX_CHARS:
        return body
    return body[:SNIPPET_MAX_CHARS].rsplit(" ", 1)[0] + "…"


def group_by_article(
    chunks: list[RankedChunk], top_k: int, include_debug: bool
) -> list[RetrievedArticle]:
    """Best chunk per article, in ranked order, cut to `top_k` articles."""
    results: list[RetrievedArticle] = []
    seen: set[str] = set()
    for chunk in chunks:
        hit = chunk.hit
        if hit.article_id in seen:
            continue
        seen.add(hit.article_id)
        results.append(
            RetrievedArticle(
                article_id=hit.article_id,
                title=hit.title,
                category=hit.category,
                score=round(hit.score, 6),
                snippet=make_snippet(chunk),
                debug=StageDebug(
                    vector_rank=chunk.vector_rank,
                    keyword_rank=chunk.keyword_rank,
                    rrf_score=None if chunk.rrf_score is None else round(chunk.rrf_score, 6),
                    rerank_score=(
                        None if chunk.rerank_score is None else round(chunk.rerank_score, 6)
                    ),
                )
                if include_debug
                else None,
            )
        )
        if len(results) == top_k:
            break
    return results


class SupportSearchService:
    def __init__(
        self,
        query_service: QueryService,
        retrieval_service: RetrievalService,
        keyword_index: KeywordIndex,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._query = query_service
        self._retrieval = retrieval_service
        self._keyword_index = keyword_index
        self._default_top_k = default_top_k

    async def search(self, request: SearchRequest) -> SupportSearchResult:
        started = time.perf_counter()
        # Cheap checks first, so a request that cannot succeed never costs an LLM call.
        if self._keyword_index.size() == 0:  # BM25 mirrors the store, and this check is free
            raise KnowledgeBaseEmptyError()
        if request.rerank and not self._retrieval.rerank_available:
            raise RerankerUnavailableError()

        # 1. Query stage
        plan = await self._query.build_query(request.question, request.history, request.query_mode)
        query_ms = (time.perf_counter() - started) * 1000

        result = SupportSearchResult(
            question=request.question,
            search_query=plan.search_query,
            query_mode=request.query_mode,
            retrieval=request.retrieval,
            rerank=request.rerank,
            needs_retrieval=plan.needs_retrieval,
            rewrite_source=plan.rewrite_source,
            rewrite_fallback=plan.rewrite_fallback,
            score_type=ScoreType.RERANK if request.rerank else _BASE_SCORE_TYPE[request.retrieval],
            results=[],
        )

        if plan.needs_retrieval:
            top_k = request.top_k or self._default_top_k
            # 2. Retrieval stage (vector / keyword / hybrid + RRF, then rerank)
            outcome = await self._retrieval.retrieve(
                plan.search_query, request.retrieval, request.rerank, min_articles=top_k
            )
            # 3. Group by article
            result.results = group_by_article(outcome.chunks, top_k, request.include_debug)
            result.rerank_fallback = outcome.rerank_fallback
            result.score_type = outcome.score_type

        logger.info(
            "search completed",
            extra={
                "question_chars": len(request.question),
                "history_items": len(request.history),
                "query_mode": request.query_mode.value,
                "retrieval": request.retrieval.value,
                "rerank": request.rerank,
                "needs_retrieval": plan.needs_retrieval,
                "rewrite_source": plan.rewrite_source,
                "rewrite_fallback": plan.rewrite_fallback,
                "rerank_fallback": result.rerank_fallback,
                "results": len(result.results),
                "query_ms": round(query_ms, 1),
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return result
