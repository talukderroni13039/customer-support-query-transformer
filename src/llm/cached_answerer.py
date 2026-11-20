"""File-backed answer cache. Wraps any `AnswerGenerator` (decorator pattern, same Protocol)."""

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from src.domains.support_search.schemas.chat_schemas import ContextChunk, GeneratedAnswer
from src.llm.base import AnswerGenerator
from src.utils.hashing import stable_hash

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_CACHE_PATH = "./data/cache/answers.json"


class CachedAnswerGenerator:
    """Returns a cached answer when the same question, history, and chunks were seen before.

    The key covers the chunk text, so editing an article invalidates answers built on it.
    Keys are hashes: the question and history are never written to disk, only the generated
    answer. Only answers that passed the grounding check are cached; failures are not.
    """

    def __init__(self, inner: AnswerGenerator, cache_path: Path, namespace: str) -> None:
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
            logger.warning("answer cache unreadable, starting empty")
            return {}
        return data if isinstance(data, dict) else {}

    def _key(
        self, question: str, history: list[str], chunks: list[ContextChunk], resolved: str
    ) -> str:
        return stable_hash(
            {
                "ns": self._namespace,
                "question": question,
                "history": history,
                "resolved": resolved,
                "chunks": [[c.label, c.chunk_id, stable_hash(c.text)] for c in chunks],
            }
        )

    async def answer(
        self,
        question: str,
        history: list[str],
        chunks: list[ContextChunk],
        resolved_question: str = "",
    ) -> GeneratedAnswer:
        key = self._key(question, history, chunks, resolved_question)
        entry = self._entries.get(key)
        if entry is not None:
            try:
                return GeneratedAnswer(
                    kind=entry["kind"],
                    answer=entry["answer"],
                    citations=entry["citations"],
                    from_cache=True,
                )
            except (KeyError, ValueError):
                logger.warning("ignoring an unreadable answer cache entry")
        result = await self._inner.answer(question, history, chunks, resolved_question)
        self._entries[key] = {
            "kind": result.kind.value,
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
        }
        await asyncio.to_thread(self._persist)
        return result

    def _persist(self) -> None:
        with self._file_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self._path)
