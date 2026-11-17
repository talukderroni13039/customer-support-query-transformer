"""ONE global exception handler. Every error leaves the API as an `ErrorResponse`."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exceptions import AppError
from src.core.middleware import REQUEST_ID_HEADER, build_meta
from src.core.schemas.api_response import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

_HTTP_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    # `input` and `ctx` are dropped on purpose: they can echo user text back or into logs.
    return [
        {"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]


async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
    headers: dict[str, str] = {}
    if isinstance(exc, AppError):
        status_code = exc.status_code
        error = ErrorDetail(code=exc.code, message=exc.message, details=exc.details)
        if status_code >= 500:
            logger.error("application error", extra={"code": exc.code})
    elif isinstance(exc, RequestValidationError):
        status_code = 422
        error = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=_validation_details(exc),
        )
    elif isinstance(exc, StarletteHTTPException):
        status_code = exc.status_code
        error = ErrorDetail(
            code=_HTTP_CODES.get(status_code, "HTTP_ERROR"), message=str(exc.detail)
        )
        headers.update(exc.headers or {})
    else:
        status_code = 500
        error = ErrorDetail(code="INTERNAL_ERROR", message=AppError.message)
        logger.error(
            "unhandled exception",
            extra={"exception_type": type(exc).__name__},
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    meta = build_meta(request)
    headers[REQUEST_ID_HEADER] = meta.request_id
    body = ErrorResponse(error=error, meta=meta)
    return JSONResponse(
        status_code=status_code, content=body.model_dump(mode="json"), headers=headers
    )


def register_exception_handlers(app: FastAPI) -> None:
    for exc_class in (AppError, RequestValidationError, StarletteHTTPException, Exception):
        app.add_exception_handler(exc_class, handle_exception)
