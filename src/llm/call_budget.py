"""Hard cap on real LLM calls, persisted on disk so it survives across runs and processes."""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from src.core.exceptions import LLMBudgetExhaustedError

DEFAULT_LLM_CALL_BUDGET = 500
DEFAULT_BUDGET_PATH = "./data/cache/llm_call_budget.json"


class LlmCallBudget:
    """Counts every real attempt (retries included) in a JSON file.

    The file is the source of truth: it is re-read on every check, so tests, the benchmark,
    and the API all share one counter. Once `limit` is reached, `consume` refuses.
    """

    def __init__(self, path: Path, limit: int = DEFAULT_LLM_CALL_BUDGET) -> None:
        self._path = path
        self._limit = limit
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def used(self) -> int:
        with self._lock:
            return self._read()

    def remaining(self) -> int:
        return max(self._limit - self.used(), 0)

    def ensure_available(self) -> None:
        if self.remaining() <= 0:
            raise LLMBudgetExhaustedError()

    def consume(self) -> int:
        """Record one real attempt. Raises `LLMBudgetExhaustedError` instead of going over."""
        with self._lock:
            used = self._read()
            if used >= self._limit:
                raise LLMBudgetExhaustedError()
            used += 1
            self._write(used)
            return used

    def _read(self) -> int:
        if not self._path.exists():
            return 0
        try:
            calls = json.loads(self._path.read_text(encoding="utf-8"))["calls"]
        except (ValueError, KeyError, TypeError) as exc:
            # Fail closed: an unreadable counter must never unlock more spending.
            raise LLMBudgetExhaustedError(
                "LLM budget exhausted (budget file is unreadable)."
            ) from exc
        if not isinstance(calls, int) or calls < 0:
            raise LLMBudgetExhaustedError("LLM budget exhausted (budget file is invalid).")
        return calls

    def _write(self, used: int) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calls": used,
            "budget": self._limit,
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)
