"""File-backed answer cache. Wraps any `AnswerGenerator` (decorator pattern, same Protocol)."""

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from src.domains.support_search.schemas.chat_schemas import AnswerContextArticle, GeneratedAnswer
from src.llm.base import AnswerGenerator
from src.utils.hashing import stable_hash

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_CACHE_PATH = "./data/cache/answers.json"


class CachedAnswerGenerator:
    """Returns a cached answer when the same question, history, and articles were seen before.

    The key covers the article text, so editing an article invalidates answers built on it.
    Keys are hashes: the question and history are never written to disk, only the generated
    answer. Failures of the inner generator are not cached.
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

    def _key(self, question: str, history: list[str], articles: list[AnswerContextArticle]) -> str:
        return stable_hash(
            {
                "ns": self._namespace,
                "question": question,
                "history": history,
                "articles": [[a.article_id, stable_hash(a.content)] for a in articles],
            }
        )

    async def answer(
        self, question: str, history: list[str], articles: list[AnswerContextArticle]
    ) -> GeneratedAnswer:
        key = self._key(question, history, articles)
        entry = self._entries.get(key)
        if entry is not None:
            return GeneratedAnswer(
                kind=entry["kind"],
                answer=entry["answer"],
                source_article_ids=entry["source_article_ids"],
                from_cache=True,
            )
        result = await self._inner.answer(question, history, articles)
        self._entries[key] = {
            "kind": result.kind.value,
            "answer": result.answer,
            "source_article_ids": result.source_article_ids,
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
