from fastapi.testclient import TestClient

from src.main import create_app


def test_health_and_docs_available() -> None:
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
