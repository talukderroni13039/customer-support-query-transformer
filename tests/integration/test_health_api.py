from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core import lifespan as lifespan_module
from src.main import create_app
from tests.fakes import FakeEmbedder


def test_health_without_lifespan_is_degraded_and_docs_available() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "degraded"
    assert body["data"]["environment"] == "test"
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_lifespan_builds_resources_and_bm25_from_chroma(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Startup wiring with a fake embedder: real PersistentClient, BM25 rebuilt from disk."""
    monkeypatch.setattr(lifespan_module, "BgeEmbedder", lambda **_: FakeEmbedder())
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "lifespan-test")
    monkeypatch.setenv("REWRITE_PROVIDER", "none")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    body = {"title": "Gift cards", "content": "Gift cards never expire and work online."}

    with TestClient(create_app()) as client:
        assert client.get("/health").json()["data"]["keyword_index_chunks"] == 0
        assert client.post("/api/v1/support/articles", json=body).status_code == 201

    with TestClient(create_app()) as client:  # restart: BM25 is rebuilt from ChromaDB
        health = client.get("/health").json()["data"]
        assert health["status"] == "ok"
        assert health["keyword_index_chunks"] == 1
        assert len(client.get("/api/v1/support/articles").json()["data"]) == 1
