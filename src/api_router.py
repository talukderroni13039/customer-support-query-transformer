"""Central /api/v1 router. Every domain router is included here."""

from fastapi import APIRouter

api_router = APIRouter(prefix="/api/v1")
