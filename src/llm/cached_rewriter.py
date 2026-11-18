"""File-backed rewrite cache. Wraps any `QueryRewriter` (decorator pattern, same Protocol)."""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from src.domains.support_search.schemas.rewrite_schemas import RewrittenQuery
from src.llm.base import QueryRewriter
from src.utils.hashing import stable_hash

logger = logging.getLogger(__name__)

DEFAULT_REWRITE_CACHE_PATH = "./data/cache/rewrites.json"


class CachedRewriter:
    """Returns a cached rewrite when the same (namespace, question, history) was seen before.

    `namespace` should identify the model and prompt version, so changing either invalidates
    old entries. Keys are hashes: the question and history themselves are never written to
    disk, only the rewritten query. Failures of the inner rewriter are not cached.
    """

    def __init__(self, inner: QueryRewriter, cache_path: Path, namespace: str) -> None:
        self._inner = inner
        self._path = cache_path
        self._namespace = namespace
        self._file_lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("rewrite cache unreadable, starting empty")
            return {}
        return data if isinstance(data, dict) else {}

    def _key(self, question: str, history: list[str]) -> str:
        return stable_hash({"ns": self._namespace, "question": question, "history": history})

    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery:
        key = self._key(question, history)
        entry = self._entries.get(key)
        if entry is not None:
            return RewrittenQuery(
                needs_retrieval=entry["needs_retrieval"],
                standalone_query=entry["standalone_query"],
                from_cache=True,
            )

        started = time.perf_counter()
        result = await self._inner.rewrite(question, history)
        self._entries[key] = {
            "needs_retrieval": result.needs_retrieval,
            "standalone_query": result.standalone_query,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        await asyncio.to_thread(self._persist)
        return result

    def cached_latency_ms(self, question: str, history: list[str]) -> float | None:
        """Latency of the original (uncached) call, for honest benchmark timings."""
        entry = self._entries.get(self._key(question, history))
        return None if entry is None else float(entry.get("latency_ms", 0.0))

    def _persist(self) -> None:
        with self._file_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self._path)
