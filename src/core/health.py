from fastapi import APIRouter

from src.core.schemas.health import HealthResult
from src.utils.env import env_str

health_router = APIRouter(tags=["health"])


@health_router.get("/health", summary="Service health")
async def health() -> HealthResult:
    return HealthResult(
        status="degraded",
        environment=env_str("APP_ENV", "dev"),
        embedding_model_loaded=False,
        reranker_loaded=False,
        vector_store="unavailable",
        keyword_index_chunks=0,
        rewrite_provider=env_str("REWRITE_PROVIDER", "anthropic"),
    )
