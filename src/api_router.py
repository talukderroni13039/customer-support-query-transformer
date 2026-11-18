"""Central /api/v1 router. Every domain router is included here."""

from fastapi import APIRouter

from src.domains.support_search.routers.api_endpoints import router as support_search_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(support_search_router)
