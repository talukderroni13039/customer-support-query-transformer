import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.core.dependencies import get_optional_resources
from src.core.lifespan import AppResources
from src.core.middleware import build_meta
from src.core.schemas.api_response import ApiResponse
from src.core.schemas.health import HealthResult
from src.utils.env import env_str

health_router = APIRouter(tags=["health"])


@health_router.get("/health", summary="Service health")
async def health(
    request: Request,
    resources: Annotated[AppResources | None, Depends(get_optional_resources)],
) -> ApiResponse[HealthResult]:
    if resources is None:
        result = HealthResult(
            status="degraded",
            environment=env_str("APP_ENV", "dev"),
            embedding_model_loaded=False,
            reranker_loaded=False,
            vector_store="unavailable",
            keyword_index_chunks=0,
            rewrite_provider="none",
        )
        return ApiResponse(data=result, meta=build_meta(request))

    try:
        await asyncio.to_thread(resources.vector_store.count)
        vector_store_ok = True
    except Exception:
        vector_store_ok = False

    result = HealthResult(
        status="ok" if vector_store_ok else "degraded",
        environment=resources.environment,
        embedding_model_loaded=True,
        reranker_loaded=False,
        vector_store="ok" if vector_store_ok else "error",
        keyword_index_chunks=resources.keyword_index.size(),
        rewrite_provider="none",
    )
    return ApiResponse(data=result, meta=build_meta(request))
