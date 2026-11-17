"""Application entry point: env loading, app creation, routers."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api_router import api_router
from src.core.health import health_router
from src.core.logging import configure_logging
from src.utils.env import env_str

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> str:
    """Load `.env.{APP_ENV}` (e.g. `.env.dev`, `.env.uat`). Real env vars always win."""
    app_env = env_str("APP_ENV", "dev")
    load_dotenv(PROJECT_ROOT / f".env.{app_env}", override=False)
    os.environ.setdefault("APP_ENV", app_env)
    return app_env


def create_app() -> FastAPI:
    configure_logging(env_str("LOG_LEVEL", "INFO"))
    app = FastAPI(
        title="Customer Support Query Transformer",
        description="Rewrite → hybrid retrieval (vector + BM25 + RRF) → rerank.",
        version="0.1.0",
    )
    app.include_router(health_router)
    app.include_router(api_router)
    return app


load_environment()
app = create_app()
