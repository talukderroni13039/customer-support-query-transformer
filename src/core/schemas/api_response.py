"""Shared response envelope. Every endpoint and every error uses it."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResponseMeta(BaseModel):
    request_id: str
    processing_time_ms: float = Field(ge=0)


class ApiResponse[T](BaseModel):
    success: Literal[True] = True
    data: T
    meta: ResponseMeta


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
    meta: ResponseMeta
