from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.dependencies import get_optional_resources
from src.core.lifespan import AppResources

BASE = "/api/v1/support/articles"
BODY = {
    "title": "Returns for Outerwear",
    "content": "## Return window\nOuterwear such as the TB-PRO-22 can be returned within 60 days."
    "\n\n## Condition requirements\nTags must be attached.",
    "category": "returns",
}


def ok(response: Any, status: int = 200) -> Any:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["success"] is True
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    return body["data"]


def err(response: Any, status: int, code: str) -> Any:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == code
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    return body["error"]


def test_article_crud_status_codes_and_envelope(client: TestClient) -> None:
    created = ok(client.post(BASE, json=BODY), 201)
    article_id = created["id"]
    assert created["chunk_count"] == 2
    assert created["content"] == BODY["content"]

    assert ok(client.get(f"{BASE}/{article_id}"))["content"] == BODY["content"]

    listed = ok(client.get(BASE))
    assert [a["id"] for a in listed] == [article_id]
    assert "content" not in listed[0]
    assert ok(client.get(BASE, params={"category": "returns"}))[0]["id"] == article_id
    assert ok(client.get(BASE, params={"category": "shipping"})) == []

    updated = ok(
        client.put(
            f"{BASE}/{article_id}",
            json={"title": "Returns v2", "content": "A single paragraph of replacement text."},
        )
    )
    assert updated["title"] == "Returns v2"
    assert updated["category"] is None
    assert updated["chunk_count"] == 1

    assert ok(client.delete(f"{BASE}/{article_id}")) == {"id": article_id, "deleted": True}
    err(client.get(f"{BASE}/{article_id}"), 404, "ARTICLE_NOT_FOUND")


def test_unknown_id_returns_404_for_get_put_delete(client: TestClient) -> None:
    error = err(client.get(f"{BASE}/KB-404"), 404, "ARTICLE_NOT_FOUND")
    assert error["details"] == {"article_id": "KB-404"}
    err(client.put(f"{BASE}/KB-404", json=BODY), 404, "ARTICLE_NOT_FOUND")
    err(client.delete(f"{BASE}/KB-404"), 404, "ARTICLE_NOT_FOUND")


def test_validation_errors_use_envelope(client: TestClient) -> None:
    cases: list[dict[str, Any]] = [
        {**BODY, "title": "ab"},
        {**BODY, "content": "too short"},
        {**BODY, "content": " " * 40},
        {**BODY, "content": "x" * 20_001},
        {**BODY, "category": "Not Valid!"},
        {**BODY, "unknown_field": 1},
        {"title": "No content"},
    ]
    for body in cases:
        err(client.post(BASE, json=body), 422, "VALIDATION_ERROR")
    err(client.get(BASE, params={"category": "BAD CATEGORY"}), 422, "VALIDATION_ERROR")
    err(client.get(f"{BASE}/bad id!"), 422, "VALIDATION_ERROR")


def test_models_not_loaded_returns_503(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_optional_resources] = lambda: None
    err(client.get(BASE), 503, "MODEL_NOT_READY")
    health = ok(client.get("/health"))
    assert health["status"] == "degraded"
    assert health["embedding_model_loaded"] is False


def test_health_reports_index_size(client: TestClient, resources: AppResources) -> None:
    ok(client.post(BASE, json=BODY), 201)
    health = ok(client.get("/health"))
    assert health["status"] == "ok"
    assert health["environment"] == "test"
    assert health["embedding_model_loaded"] is True
    assert health["vector_store"] == "ok"
    assert health["keyword_index_chunks"] == 2 == resources.keyword_index.size()


def test_health_degraded_when_vector_store_fails(
    client: TestClient, resources: AppResources, monkeypatch: Any
) -> None:
    def broken() -> int:
        raise RuntimeError("chroma down")

    monkeypatch.setattr(resources.vector_store, "count", broken)
    health = ok(client.get("/health"))
    assert health["status"] == "degraded"
    assert health["vector_store"] == "error"
