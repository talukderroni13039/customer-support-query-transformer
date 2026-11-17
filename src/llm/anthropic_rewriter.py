"""Query rewriter: Instructor + AsyncAnthropic, guarded by the LLM call budget."""

import logging
import time
from typing import Any

import anthropic
import httpx
import instructor

from src.core.exceptions import LLMBudgetExhaustedError, LLMError, LLMTimeoutError
from src.domains.support_search.prompts.rewrite_prompt import (
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_user_message,
)
from src.domains.support_search.schemas.rewrite_schemas import RewrittenQuery
from src.llm.call_budget import LlmCallBudget

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "claude-haiku-5-5"
DEFAULT_LLM_MAX_TOKENS = 300
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_TIMEOUT_SECONDS = 10.0
# SDK-level retries for transient failures (429, 5xx, overloaded, connection errors).
DEFAULT_TRANSPORT_RETRIES = 1


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


class AnthropicRewriter:
    """Implements `QueryRewriter`.

    Every HTTP request sent to Anthropic passes through `_count_request`, which records one
    attempt in the budget and refuses once the budget is spent. That covers both kinds of
    retry: Instructor's (validation failures) and the SDK's own (429, 5xx, overloaded,
    connection errors). Nothing is ever sent uncounted.
    """

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
        self._budget = budget
        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        # A request hook (not a transport) so the count also covers proxy-mounted transports.
        http_client = anthropic.DefaultAsyncHttpxClient(
            event_hooks={"request": [self._count_request]},
            transport=transport,  # None → the SDK default; tests pass an httpx.MockTransport
        )
        self._anthropic = anthropic.AsyncAnthropic(
            api_key=api_key,
            max_retries=transport_retries,
            timeout=timeout_seconds,
            http_client=http_client,
        )
        self._client: Any = instructor.from_anthropic(self._anthropic)

    async def _count_request(self, request: httpx.Request) -> None:
        used = self._budget.consume()  # raises LLMBudgetExhaustedError at the cap
        logger.info(
            "llm call", extra={"llm_calls_used": used, "llm_call_budget": self._budget.limit}
        )

    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery:
        self._budget.ensure_available()  # refuse before building any request
        started = time.perf_counter()
        try:
            result: RewrittenQuery = await self._client.create(
                response_model=RewrittenQuery,
                max_retries=self._max_retries,
                model=self.model,
                max_tokens=self._max_tokens,
                system=REWRITE_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": build_rewrite_user_message(question, history)}
                ],
            )
        except Exception as exc:
            chain = _exception_chain(exc)
            status_code = next(
                (e.status_code for e in chain if isinstance(e, anthropic.APIStatusError)), None
            )
            # Type names and status only: exception messages can echo prompt or model text.
            cause = " <- ".join(dict.fromkeys(type(e).__name__ for e in chain))
            logger.warning(
                "rewrite failed", extra={"exception_types": cause, "status_code": status_code}
            )
            if any(isinstance(e, LLMBudgetExhaustedError) for e in chain):
                raise LLMBudgetExhaustedError() from exc
            if any(
                isinstance(e, anthropic.APITimeoutError | httpx.TimeoutException) for e in chain
            ):
                raise LLMTimeoutError() from exc
            status = f", HTTP {status_code}" if status_code else ""
            raise LLMError(
                f"The query rewrite provider returned an error ({cause}{status}).",
                details={"cause": cause, "status_code": status_code},
            ) from exc

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
        await self._anthropic.close()
