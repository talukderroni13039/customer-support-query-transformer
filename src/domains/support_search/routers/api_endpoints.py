"""HTTP layer for the support_search domain: receive, call a service, wrap in `ApiResponse`."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request, status

from src.core.dependencies import get_article_service
from src.core.middleware import build_meta
from src.core.schemas.api_response import ApiResponse, ErrorResponse
from src.domains.support_search.schemas.article_schemas import (
    ArticleCreate,
    ArticleDeleteResult,
    ArticleResult,
    ArticleSummary,
    ArticleUpdate,
)
from src.domains.support_search.services.article_service import ArticleService

router = APIRouter(
    prefix="/support",
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)

ArticleServiceDep = Annotated[ArticleService, Depends(get_article_service)]
ArticleId = Annotated[
    str, Path(pattern=r"^[A-Za-z0-9._-]{1,64}$", description="Article ID", examples=["KB-012"])
]
NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse}}


@router.post(
    "/articles",
    status_code=status.HTTP_201_CREATED,
    tags=["articles"],
    summary="Create an article (chunk, embed, index in ChromaDB and BM25)",
)
async def create_article(
    payload: ArticleCreate, request: Request, service: ArticleServiceDep
) -> ApiResponse[ArticleResult]:
    return ApiResponse(data=await service.create(payload), meta=build_meta(request))


@router.get("/articles", tags=["articles"], summary="List articles")
async def list_articles(
    request: Request,
    service: ArticleServiceDep,
    category: Annotated[
        str | None, Query(max_length=50, pattern=r"^[a-z0-9-]+$", description="Filter by category")
    ] = None,
) -> ApiResponse[list[ArticleSummary]]:
    return ApiResponse(data=await service.list_articles(category), meta=build_meta(request))


@router.get(
    "/articles/{article_id}",
    tags=["articles"],
    summary="Get one article with its full content",
    responses=NOT_FOUND,
)
async def get_article(
    article_id: ArticleId, request: Request, service: ArticleServiceDep
) -> ApiResponse[ArticleResult]:
    return ApiResponse(data=await service.get(article_id), meta=build_meta(request))


@router.put(
    "/articles/{article_id}",
    tags=["articles"],
    summary="Replace an article and re-index it",
    responses=NOT_FOUND,
)
async def update_article(
    article_id: ArticleId, payload: ArticleUpdate, request: Request, service: ArticleServiceDep
) -> ApiResponse[ArticleResult]:
    return ApiResponse(data=await service.update(article_id, payload), meta=build_meta(request))


@router.delete(
    "/articles/{article_id}",
    tags=["articles"],
    summary="Delete an article from both indexes",
    responses=NOT_FOUND,
)
async def delete_article(
    article_id: ArticleId, request: Request, service: ArticleServiceDep
) -> ApiResponse[ArticleDeleteResult]:
    return ApiResponse(data=await service.delete(article_id), meta=build_meta(request))
