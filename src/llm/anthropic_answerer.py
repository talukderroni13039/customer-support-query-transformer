"""Grounded answer generator: Instructor + AsyncAnthropic, guarded by the LLM call budget."""

import logging
import time

import httpx

from src.domains.support_search.prompts.answer_prompt import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_message,
)
from src.domains.support_search.schemas.chat_schemas import ContextChunk, GeneratedAnswer
from src.llm.anthropic_client import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TRANSPORT_RETRIES,
    BudgetedAnthropicClient,
)
from src.llm.call_budget import LlmCallBudget

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_MAX_TOKENS = 900
DEFAULT_ANSWER_TIMEOUT_SECONDS = 20.0
# One correction attempt when the answer fails the schema or the grounding check.
DEFAULT_ANSWER_MAX_RETRIES = 1


class AnthropicAnswerGenerator:
    """Implements `AnswerGenerator`.

    The chunks are handed to the response model as validation context, so `GeneratedAnswer`
    runs the grounding check on Claude's output. A failed check goes back to Claude once as a
    correction request; a second failure raises `LLMOutputInvalidError`. No sampling
    parameters are sent: this model family rejects a non-default temperature.
    """

    def __init__(
        self,
        api_key: str,
        budget: LlmCallBudget,
        *,
        model: str = DEFAULT_LLM_MODEL,
        max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
        max_retries: int = DEFAULT_ANSWER_MAX_RETRIES,
        timeout_seconds: float = DEFAULT_ANSWER_TIMEOUT_SECONDS,
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

    async def answer(
        self,
        question: str,
        history: list[str],
        chunks: list[ContextChunk],
        resolved_question: str = "",
    ) -> GeneratedAnswer:
        started = time.perf_counter()
        result = await self._client.structured(
            GeneratedAnswer,
            system=ANSWER_SYSTEM_PROMPT,
            user_message=build_answer_user_message(question, history, chunks, resolved_question),
            max_tokens=self._max_tokens,
            operation="answer",
            context={
                "chunks": {c.label: c.checkable_text for c in chunks},
                "question_text": question,
                "history_text": " ".join(history),
                "titles_text": " ".join(c.title for c in chunks),
            },
        )
        logger.info(
            "answer completed",
            extra={
                "model": self.model,
                "kind": result.kind.value,
                "chunks": len(chunks),
                "citations": len(result.citations),
                "history_items": len(history),
                "question_chars": len(question),
                "answer_chars": len(result.answer),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return result.model_copy(update={"from_cache": False})

    async def aclose(self) -> None:
        await self._client.aclose()
