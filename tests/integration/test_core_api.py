"""Envelope, global exception handler, and request-ID middleware."""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from src.core.exceptions import ArticleNotFoundError, ModelNotReadyError
from src.core.logging import JsonFormatter, request_id_var
from src.main import create_app


class _Body(BaseModel):
    name: str = Field(min_length=3)


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    @app.post("/_test/echo")
    async def echo(body: _Body) -> dict[str, str]:
        return {"name": body.name}

    @app.get("/_test/not-found")
    async def not_found() -> None:
        raise ArticleNotFoundError(details={"article_id": "KB-404"})

    @app.get("/_test/not-ready")
    async def not_ready() -> None:
        raise ModelNotReadyError()

    @app.get("/_test/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _assert_error(response, status: int, code: str) -> dict:  # type: ignore[no-untyped-def, type-arg]
    assert response.status_code == status
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == code
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    assert body["meta"]["processing_time_ms"] >= 0
    return body  # type: ignore[no-any-return]


def test_validation_error_uses_envelope_without_echoing_input(client: TestClient) -> None:
    body = _assert_error(client.post("/_test/echo", json={"name": "x"}), 422, "VALIDATION_ERROR")
    detail = body["error"]["details"][0]
    assert detail["loc"] == ["body", "name"]
    assert "input" not in detail


def test_app_error_uses_envelope(client: TestClient) -> None:
    body = _assert_error(client.get("/_test/not-found"), 404, "ARTICLE_NOT_FOUND")
    assert body["error"]["details"] == {"article_id": "KB-404"}
    _assert_error(client.get("/_test/not-ready"), 503, "MODEL_NOT_READY")


def test_unhandled_exception_is_generic_500(client: TestClient) -> None:
    body = _assert_error(client.get("/_test/boom"), 500, "INTERNAL_ERROR")
    assert "secret" not in json.dumps(body)


def test_unknown_route_and_wrong_method_use_envelope(client: TestClient) -> None:
    _assert_error(client.get("/nope"), 404, "NOT_FOUND")
    _assert_error(client.post("/health"), 405, "METHOD_NOT_ALLOWED")


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    given = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert given.headers["X-Request-ID"] == "abc-123"
    assert given.json()["meta"]["request_id"] == "abc-123"
    generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces!"})
    assert len(generated.headers["X-Request-ID"]) == 32


def test_json_formatter_includes_request_id_extra_and_exception() -> None:
    token = request_id_var.set("req-1")
    try:
        try:
            raise ValueError("bad")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "t", logging.ERROR, __file__, 1, "msg %s", ("x",), sys.exc_info()
            )
        record.custom = 5
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "msg x"
    assert payload["request_id"] == "req-1"
    assert payload["custom"] == 5
    assert "ValueError" in payload["exception"]
