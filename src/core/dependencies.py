"""ALL dependency-injection providers. Routers use `Depends(...)` on these and nothing else."""

from typing import Annotated

from fastapi import Depends, Request

from src.core.exceptions import ModelNotReadyError
from src.core.lifespan import AppResources
from src.domains.support_search.chunking.article_chunker import DEFAULT_CHUNK_MAX_CHARS
from src.domains.support_search.services.article_service import ArticleService
from src.domains.support_search.services.query_service import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    QueryService,
)
from src.domains.support_search.services.retrieval_service import (
    DEFAULT_CANDIDATES_PER_RETRIEVER,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_MIN_SCORE,
    RetrievalService,
)
from src.domains.support_search.services.support_search_service import (
    DEFAULT_TOP_K,
    SupportSearchService,
)
from src.embeddings.base import Embedder
from src.keyword_search.base import KeywordIndex
from src.llm.base import QueryRewriter
from src.reranking.base import Reranker
from src.utils.env import env_bool, env_float, env_int
from src.utils.rank_fusion import DEFAULT_RRF_K
from src.vectorstore.base import VectorStore


def get_optional_resources(request: Request) -> AppResources | None:
    resources: AppResources | None = getattr(request.app.state, "resources", None)
    return resources


def get_resources(
    resources: Annotated[AppResources | None, Depends(get_optional_resources)],
) -> AppResources:
    if resources is None:
        raise ModelNotReadyError()
    return resources


ResourcesDep = Annotated[AppResources, Depends(get_resources)]


def get_embedder(resources: ResourcesDep) -> Embedder:
    return resources.embedder


def get_vector_store(resources: ResourcesDep) -> VectorStore:
    return resources.vector_store


def get_keyword_index(resources: ResourcesDep) -> KeywordIndex:
    return resources.keyword_index


def get_reranker(resources: ResourcesDep) -> Reranker | None:
    """None when reranking is disabled (RERANK_ENABLED=false)."""
    return resources.reranker


def get_rewriter(resources: ResourcesDep) -> QueryRewriter | None:
    """None when the rewrite provider is disabled or has no API key."""
    return resources.rewriter


def get_article_service(
    resources: ResourcesDep,
    embedder: Annotated[Embedder, Depends(get_embedder)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
) -> ArticleService:
    return ArticleService(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        write_lock=resources.index_write_lock,
        chunk_max_chars=env_int("CHUNK_MAX_CHARS", DEFAULT_CHUNK_MAX_CHARS),
    )


def get_retrieval_service(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
) -> RetrievalService:
    return RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        reranker=reranker,
        candidates_per_retriever=env_int(
            "CANDIDATES_PER_RETRIEVER", DEFAULT_CANDIDATES_PER_RETRIEVER
        ),
        rrf_k=env_int("RRF_K", DEFAULT_RRF_K),
        rerank_candidates=env_int("RERANK_CANDIDATES", DEFAULT_RERANK_CANDIDATES),
        rerank_min_score=env_float("RERANK_MIN_SCORE", DEFAULT_RERANK_MIN_SCORE),
    )


def get_query_service(
    rewriter: Annotated[QueryRewriter | None, Depends(get_rewriter)],
) -> QueryService:
    return QueryService(
        rewriter=rewriter,
        max_history_messages=env_int("MAX_HISTORY_MESSAGES", DEFAULT_MAX_HISTORY_MESSAGES),
        strict=env_bool("REWRITE_STRICT", False),
    )


def get_support_search_service(
    query_service: Annotated[QueryService, Depends(get_query_service)],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
) -> SupportSearchService:
    return SupportSearchService(
        query_service=query_service,
        retrieval_service=retrieval_service,
        keyword_index=keyword_index,
        default_top_k=env_int("DEFAULT_TOP_K", DEFAULT_TOP_K),
    )
