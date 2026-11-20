"""Chat: retrieve, gate on relevance, write ONE grounded answer with citations, guard output.

1. rewrite the message into a standalone query (QueryService)
2. hybrid retrieval + rerank (RetrievalService)
3. threshold gate: keep the top chunks whose rerank score clears the minimum
4. Claude answers from those chunks only, citing a chunk and an exact quote per fact
5. grounding check (inside the answer generator; one correction attempt)
6. output guardrail: mask contact details and links that are not in the cited chunks
"""

import logging
import time

from src.core.exceptions import KnowledgeBaseEmptyError, LLMOutputInvalidError
from src.domains.support_search.schemas.chat_schemas import (
    AnswerKind,
    ChatCitation,
    ChatRequest,
    ChatResult,
    ChatSource,
    ContextChunk,
    GeneratedAnswer,
)
from src.domains.support_search.schemas.search_schemas import (
    QueryMode,
    RetrievalMode,
    ScoreType,
)
from src.domains.support_search.services.query_service import QueryService
from src.domains.support_search.services.retrieval_service import RetrievalService
from src.keyword_search.base import KeywordIndex
from src.llm.base import AnswerGenerator
from src.utils.output_guard import apply_output_guardrail

logger = logging.getLogger(__name__)

# Calibrated with scripts/calibrate_threshold.py (see docs/BENCHMARK.md): off-topic questions
# score at most 0.004, uncovered support questions at most 0.115, and 29 of 30 correct answers
# score above 0.1. A cut-off of 0.5 would have discarded 13% of correct answers.
DEFAULT_CHAT_MIN_RERANK_SCORE = 0.05
DEFAULT_CHAT_CONTEXT_CHUNKS = 5
MIN_ARTICLES = 3

OFF_TOPIC_MESSAGE = (
    "I can only help with Northwind Gear support: orders, shipping, returns and refunds, "
    "exchanges, warranty, sizing, payment, your account, and our products. "
    "That question is outside what I can answer here. Is there something about your order "
    "or our gear I can help with?"
)
NOT_FOUND_MESSAGE = (
    "I don't have that information in our help articles, so I don't want to guess. "
    "Please contact our support team and they'll look into it for you. "
    "If you can tell me a bit more (for example the product name or what you're trying to "
    "do), I can try again."
)
SMALL_TALK_MESSAGE = "You're welcome! Let me know if there's anything else I can help with."
FALLBACK_INTRO = "I couldn't write a full answer right now, but this help article looks relevant:"


class SupportChatService:
    """Always uses the full pipeline (rewrite → hybrid → rerank); the caller has no switches."""

    def __init__(
        self,
        query_service: QueryService,
        retrieval_service: RetrievalService,
        keyword_index: KeywordIndex,
        answer_generator: AnswerGenerator | None,
        min_rerank_score: float = DEFAULT_CHAT_MIN_RERANK_SCORE,
        max_context_chunks: int = DEFAULT_CHAT_CONTEXT_CHUNKS,
    ) -> None:
        self._query = query_service
        self._retrieval = retrieval_service
        self._keyword_index = keyword_index
        self._answerer = answer_generator
        self._min_score = min_rerank_score
        self._max_chunks = max_context_chunks

    async def chat(self, request: ChatRequest) -> ChatResult:
        started = time.perf_counter()
        if self._keyword_index.size() == 0:  # cheap check before any LLM call
            raise KnowledgeBaseEmptyError()

        plan = await self._query.build_query(request.question, request.history, QueryMode.REWRITE)
        result = ChatResult(
            question=request.question,
            answer="",
            kind=AnswerKind.ANSWER,
            search_query=plan.search_query,
            rewrite_fallback=plan.rewrite_fallback,
        )
        candidates = 0
        chunks: list[ContextChunk] = []
        if not plan.needs_retrieval:
            # The rewriter already saw the conversation and found no question in it.
            result.kind, result.answer = AnswerKind.SMALL_TALK, SMALL_TALK_MESSAGE
        else:
            chunks, candidates = await self._relevant_chunks(plan.search_query)
            await self._write_answer(request, chunks, result)

        logger.info(
            "chat completed",
            extra={
                "kind": result.kind.value,
                "candidate_chunks": candidates,
                "chunks_past_gate": len(chunks),
                "citations": len(result.citations),
                "question_chars": len(request.question),
                "history_items": len(request.history),
                "answer_chars": len(result.answer),
                "rewrite_fallback": result.rewrite_fallback,
                "answer_fallback": result.answer_fallback,
                "grounding_failed": result.grounding_failed,
                "guardrail_actions": result.guardrail_actions,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return result

    async def _relevant_chunks(self, search_query: str) -> tuple[list[ContextChunk], int]:
        """Retrieve, rerank, and apply the threshold gate. Returns (kept chunks, candidates)."""
        outcome = await self._retrieval.retrieve(
            search_query,
            RetrievalMode.HYBRID,
            rerank=self._retrieval.rerank_available,
            min_articles=MIN_ARTICLES,
        )
        ranked = outcome.chunks
        if outcome.score_type is ScoreType.RERANK:
            # Only rerank scores are calibrated (0-1). Without them (reranker off or failed)
            # there is nothing trustworthy to gate on, so the top chunks pass as they are.
            ranked = [c for c in ranked if c.hit.score >= self._min_score]
        kept = [
            ContextChunk(
                label=f"C{i}",
                chunk_id=c.hit.chunk_id,
                article_id=c.hit.article_id,
                title=c.hit.title,
                heading=c.hit.heading,
                text=c.hit.text,
                score=c.hit.score,
            )
            for i, c in enumerate(ranked[: self._max_chunks], start=1)
        ]
        return kept, len(outcome.chunks)

    async def _write_answer(
        self, request: ChatRequest, chunks: list[ContextChunk], result: ChatResult
    ) -> None:
        if self._answerer is None:
            self._fallback(chunks, result)
            return
        try:
            generated = await self._answerer.answer(
                request.question, self._query.recent_history(request.history), chunks
            )
        except LLMOutputInvalidError:
            # The answer failed the grounding check twice: never show an unverified answer.
            result.kind, result.answer = AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE
            result.grounding_failed = True
            return
        except Exception as exc:
            # Same resilience rule as the rewrite: an LLM outage never fails the request.
            logger.warning(
                "answer generation failed, quoting the best chunk",
                extra={"exception_type": type(exc).__name__},
            )
            self._fallback(chunks, result)
            return

        result.kind = generated.kind
        if generated.kind is AnswerKind.OFF_TOPIC:
            result.answer = OFF_TOPIC_MESSAGE
        elif generated.kind is AnswerKind.NOT_FOUND:
            result.answer = NOT_FOUND_MESSAGE
        elif generated.kind is AnswerKind.SMALL_TALK:
            result.answer = generated.answer
        else:
            self._finish_answer(generated, chunks, result)

    @staticmethod
    def _finish_answer(
        generated: GeneratedAnswer, chunks: list[ContextChunk], result: ChatResult
    ) -> None:
        by_label = {c.label: c for c in chunks}
        # Defence in depth: the generator already verified these, a cache entry may be older.
        cited = [
            (by_label[c.chunk_id], c.quote) for c in generated.citations if c.chunk_id in by_label
        ]
        if not cited:
            result.kind, result.answer = AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE
            result.grounding_failed = True
            return

        allowed_text = " ".join(chunk.checkable_text for chunk, _ in cited)
        result.answer, result.guardrail_actions = apply_output_guardrail(
            generated.answer, allowed_text
        )
        result.citations = [
            ChatCitation(
                article_id=chunk.article_id, title=chunk.title, heading=chunk.heading, quote=quote
            )
            for chunk, quote in cited
        ]
        seen: dict[str, str] = {}
        for chunk, _ in cited:
            seen.setdefault(chunk.article_id, chunk.title)
        result.sources = [ChatSource(article_id=i, title=t) for i, t in seen.items()]

    @staticmethod
    def _fallback(chunks: list[ContextChunk], result: ChatResult) -> None:
        result.answer_fallback = True
        if not chunks:
            result.kind, result.answer = AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE
            return
        best = chunks[0]
        result.kind = AnswerKind.ANSWER
        result.answer = f"{FALLBACK_INTRO}\n\n{best.source}\n{best.body}"
        result.sources = [ChatSource(article_id=best.article_id, title=best.title)]
        result.citations = [
            ChatCitation(
                article_id=best.article_id, title=best.title, heading=best.heading, quote=best.body
            )
        ]
