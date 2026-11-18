"""Structured JSON logging with a per-request ID.

Never pass questions, history, or article content to a logger at INFO or above;
log lengths, modes, counts, and timings instead.
"""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

# Third-party loggers that may echo prompt or model text in their messages.
_NOISY_LOGGERS = (
    "instructor",
    "httpx",
    "httpcore",
    "anthropic",
    "chromadb",
    "sentence_transformers",
)


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object, including `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler on the root logger (idempotent)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.CRITICAL if name == "instructor" else logging.WARNING
        )
