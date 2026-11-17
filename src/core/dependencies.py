"""ALL dependency-injection providers. Routers use `Depends(...)` on these and nothing else."""

from typing import Annotated

from fastapi import Depends, Request

from src.core.exceptions import ModelNotReadyError
from src.core.lifespan import AppResources
from src.domains.support_search.chunking.article_chunker import DEFAULT_CHUNK_MAX_CHARS
from src.domains.support_search.services.article_service import ArticleService
from src.domains.support_search.services.retrieval_service import (
    DEFAULT_CANDIDATES_PER_RETRIEVER,
    RetrievalService,
)
from src.domains.support_search.services.support_search_service import (
    DEFAULT_TOP_K,
    SupportSearchService,
)
from src.embeddings.base import Embedder
from src.keyword_search.base import KeywordIndex
from src.utils.env import env_int
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
) -> RetrievalService:
    return RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        candidates_per_retriever=env_int(
            "CANDIDATES_PER_RETRIEVER", DEFAULT_CANDIDATES_PER_RETRIEVER
        ),
        rrf_k=env_int("RRF_K", DEFAULT_RRF_K),
    )


def get_support_search_service(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    keyword_index: Annotated[KeywordIndex, Depends(get_keyword_index)],
) -> SupportSearchService:
    return SupportSearchService(
        retrieval_service=retrieval_service,
        keyword_index=keyword_index,
        default_top_k=env_int("DEFAULT_TOP_K", DEFAULT_TOP_K),
    )
