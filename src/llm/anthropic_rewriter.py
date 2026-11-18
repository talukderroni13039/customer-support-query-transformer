"""Query rewriter: Instructor + AsyncAnthropic, guarded by the LLM call budget."""

import logging
import time

import httpx

from src.domains.support_search.prompts.rewrite_prompt import (
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_user_message,
)
from src.domains.support_search.schemas.rewrite_schemas import RewrittenQuery
from src.llm.anthropic_client import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_TRANSPORT_RETRIES,
    BudgetedAnthropicClient,
)
from src.llm.call_budget import LlmCallBudget

logger = logging.getLogger(__name__)

DEFAULT_LLM_MAX_TOKENS = 300

__all__ = [
    "DEFAULT_LLM_MAX_RETRIES",
    "DEFAULT_LLM_MAX_TOKENS",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_TIMEOUT_SECONDS",
    "AnthropicRewriter",
]


class AnthropicRewriter:
    """Implements `QueryRewriter`. Every HTTP attempt is counted by `BudgetedAnthropicClient`."""

    def __init__(
        self,
        api_key: str,
        budget: LlmCallBudget,
        *,
        model: str = DEFAULT_LLM_MODEL,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._client = BudgetedAnthropicClient(
            api_key,
            budget,
            model=model,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            transport_retries=transport_retries,
            transport=transport,
        )

    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery:
        started = time.perf_counter()
        result = await self._client.structured(
            RewrittenQuery,
            system=REWRITE_SYSTEM_PROMPT,
            user_message=build_rewrite_user_message(question, history),
            max_tokens=self._max_tokens,
            operation="rewrite",
        )
        logger.info(
            "rewrite completed",
            extra={
                "model": self.model,
                "history_items": len(history),
                "question_chars": len(question),
                "needs_retrieval": result.needs_retrieval,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return result.model_copy(update={"from_cache": False})

    async def aclose(self) -> None:
        await self._client.aclose()
