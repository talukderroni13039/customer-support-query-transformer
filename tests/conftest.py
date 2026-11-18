"""Shared test setup. Default tests never touch real APIs or real models."""

import os

# Must run before `src.main` is imported so no real `.env.*` file (and no API key) is loaded.
# Live tests load the key themselves, inside their own fixture.
os.environ["APP_ENV"] = "test"
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.dependencies import get_optional_resources
from src.core.lifespan import AppResources
from src.keyword_search.bm25_index import BM25Index
from src.main import create_app
from tests.fakes import (
    FakeAnswerGenerator,
    FakeEmbedder,
    FakeReranker,
    FakeRewriter,
    make_chroma_store,
)


@pytest.fixture
def resources() -> AppResources:
    """Fakes for every model and for Claude, in-memory Chroma, and the real BM25 index."""
    return AppResources(
        environment="test",
        embedder=FakeEmbedder(),
        vector_store=make_chroma_store(),
        keyword_index=BM25Index(),
        reranker=FakeReranker(),
        rewriter=FakeRewriter(),
        rewrite_provider="fake",
        answer_generator=FakeAnswerGenerator(),
    )


@pytest.fixture
def app(resources: AppResources) -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_optional_resources] = lambda: resources
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    # No `with`: the lifespan (real model loading) never runs in default tests.
    return TestClient(app, raise_server_exceptions=False)
