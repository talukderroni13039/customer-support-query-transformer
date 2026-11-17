from fastapi import APIRouter, Request

from src.core.middleware import build_meta
from src.core.schemas.api_response import ApiResponse
from src.core.schemas.health import HealthResult
from src.utils.env import env_str

health_router = APIRouter(tags=["health"])


@health_router.get("/health", summary="Service health")
async def health(request: Request) -> ApiResponse[HealthResult]:
    result = HealthResult(
        status="degraded",
        environment=env_str("APP_ENV", "dev"),
        embedding_model_loaded=False,
        reranker_loaded=False,
        vector_store="unavailable",
        keyword_index_chunks=0,
        rewrite_provider=env_str("REWRITE_PROVIDER", "anthropic"),
    )
    return ApiResponse(data=result, meta=build_meta(request))
