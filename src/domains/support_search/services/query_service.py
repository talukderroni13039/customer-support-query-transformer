"""Query stage: question + history → the search query, per `query_mode`."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.core.exceptions import AppError, LLMError
from src.domains.support_search.schemas.search_schemas import (
    HistoryItem,
    HistoryMessage,
    QueryMode,
)
from src.llm.base import QueryRewriter

logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_MESSAGES = 12

_ROLE_LABELS = {"customer": "Customer", "agent": "Agent"}


def history_text(item: HistoryItem) -> str:
    """The words of a history item, without the speaker."""
    return (item.text if isinstance(item, HistoryMessage) else item).strip()


def render_history_item(item: HistoryItem) -> str:
    """A history item as shown to the LLM: `Customer: ...` / `Agent: ...` when the role is known."""
    text = history_text(item)
    if isinstance(item, HistoryMessage):
        return f"{_ROLE_LABELS[item.role]}: {text}"
    return text


@dataclass(frozen=True)
class QueryPlan:
    search_query: str
    needs_retrieval: bool = True
    # none (baseline mode) | skipped (no history) | llm | cache | fallback | disabled
    rewrite_source: str = "none"
    rewrite_fallback: bool = False


class QueryService:
    def __init__(
        self,
        rewriter: QueryRewriter | None,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
        strict: bool = False,
    ) -> None:
        self._rewriter = rewriter
        self._max_history = max_history_messages
        self._strict = strict

    def _recent(self, history: Sequence[HistoryItem]) -> list[HistoryItem]:
        """Drop blank items, then keep only the last `max_history_messages`."""
        kept = [item for item in history if history_text(item)]
        return kept[-self._max_history :] if self._max_history > 0 else []

    def recent_history(self, history: Sequence[HistoryItem]) -> list[str]:
        """The recent history as lines for an LLM prompt, with speaker labels where known."""
        return [render_history_item(item) for item in self._recent(history)]

    async def build_query(
        self, question: str, history: Sequence[HistoryItem], mode: QueryMode
    ) -> QueryPlan:
        if mode is QueryMode.QUESTION_ONLY:
            return QueryPlan(search_query=question)

        if mode is QueryMode.FULL_HISTORY:
            # A search query, not a prompt: speaker labels would only add noise.
            texts = [history_text(item) for item in self._recent(history)]
            return QueryPlan(search_query="\n".join([*texts, question]))

        recent = self.recent_history(history)

        # QueryMode.REWRITE
        if not recent:
            return QueryPlan(search_query=question, rewrite_source="skipped")
        if self._rewriter is None:
            return QueryPlan(
                search_query=question, rewrite_source="disabled", rewrite_fallback=True
            )

        try:
            rewritten = await self._rewriter.rewrite(question, recent)
        except Exception as exc:
            logger.warning(
                "rewrite failed, falling back to the question",
                extra={"exception_type": type(exc).__name__, "strict": self._strict},
            )
            if self._strict:
                if isinstance(exc, AppError):
                    raise
                raise LLMError() from exc
            return QueryPlan(
                search_query=question, rewrite_source="fallback", rewrite_fallback=True
            )

        source = "cache" if rewritten.from_cache else "llm"
        if not rewritten.needs_retrieval:
            return QueryPlan(search_query="", needs_retrieval=False, rewrite_source=source)
        return QueryPlan(search_query=rewritten.standalone_query, rewrite_source=source)
