"""Request ID + start time middleware (pure ASGI)."""

import logging
import re
import time
import uuid

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.core.logging import request_id_var
from src.core.schemas.api_response import ResponseMeta

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Assigns a request ID, records the start time, echoes `X-Request-ID`, logs the request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Request(scope).headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        start = time.perf_counter()
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["start_time"] = start
        token = request_id_var.set(request_id)
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != REQUEST_ID_HEADER.lower().encode()
                ]
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request completed",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_var.reset(token)


def build_meta(request: Request) -> ResponseMeta:
    """Build the envelope `meta` from what the middleware stored on the request."""
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    start = getattr(request.state, "start_time", None)
    elapsed_ms = (time.perf_counter() - start) * 1000 if start is not None else 0.0
    return ResponseMeta(request_id=request_id, processing_time_ms=round(elapsed_ms, 2))
