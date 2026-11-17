from fastapi.testclient import TestClient

from src.main import create_app


def test_health_uses_envelope_and_docs_available() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["environment"] == "test"
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    assert client.get("/docs").status_code == 200
