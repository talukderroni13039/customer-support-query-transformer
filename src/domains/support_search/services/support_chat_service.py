"""Chat: search the knowledge base, then write ONE grounded answer (or a clear non-answer)."""

import asyncio
import logging
import time

from src.domains.support_search.schemas.chat_schemas import (
    AnswerContextArticle,
    AnswerKind,
    ChatRequest,
    ChatResult,
    ChatSource,
)
from src.domains.support_search.schemas.search_schemas import (
    QueryMode,
    RetrievalMode,
    RetrievedArticle,
    SearchRequest,
)
from src.domains.support_search.services.query_service import QueryService
from src.domains.support_search.services.support_search_service import SupportSearchService
from src.llm.base import AnswerGenerator
from src.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

CONTEXT_ARTICLES = 3
MAX_ARTICLE_CONTEXT_CHARS = 6000

OFF_TOPIC_MESSAGE = (
    "I can only help with Northwind Gear support: orders, shipping, returns and refunds, "
    "exchanges, warranty, sizing, payment, your account, and our products. "
    "That question is outside what I can answer here. Is there something about your order "
    "or our gear I can help with?"
)
NOT_FOUND_MESSAGE = (
    "I couldn't find that in our help articles, so I don't want to guess. "
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
        search_service: SupportSearchService,
        query_service: QueryService,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator | None,
        rerank_available: bool,
    ) -> None:
        self._search = search_service
        self._query = query_service
        self._vector_store = vector_store
        self._answerer = answer_generator
        self._rerank = rerank_available

    async def chat(self, request: ChatRequest) -> ChatResult:
        started = time.perf_counter()
        search = await self._search.search(
            SearchRequest(
                question=request.question,
                history=request.history,
                top_k=CONTEXT_ARTICLES,
                query_mode=QueryMode.REWRITE,
                retrieval=RetrievalMode.HYBRID,
                rerank=self._rerank,
            )
        )
        result = ChatResult(
            question=request.question,
            answer="",
            kind=AnswerKind.ANSWER,
            search_query=search.search_query,
            rewrite_fallback=search.rewrite_fallback,
        )

        if not search.needs_retrieval:
            # The rewriter already saw the conversation and found no question in it.
            result.kind, result.answer = AnswerKind.SMALL_TALK, SMALL_TALK_MESSAGE
        else:
            articles = await self._load_articles(search.results)
            await self._write_answer(request, articles, search.results, result)

        logger.info(
            "chat completed",
            extra={
                "kind": result.kind.value,
                "sources": len(result.sources),
                "question_chars": len(request.question),
                "history_items": len(request.history),
                "answer_chars": len(result.answer),
                "rewrite_fallback": result.rewrite_fallback,
                "answer_fallback": result.answer_fallback,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return result

    async def _load_articles(self, hits: list[RetrievedArticle]) -> list[AnswerContextArticle]:
        """Full text of the retrieved articles: the answer must not be limited to a snippet."""
        articles: list[AnswerContextArticle] = []
        for hit in hits:
            chunks = await asyncio.to_thread(self._vector_store.get_article_chunks, hit.article_id)
            content = "".join(chunk.text for chunk in chunks)[:MAX_ARTICLE_CONTEXT_CHARS]
            if content.strip():
                articles.append(
                    AnswerContextArticle(
                        article_id=hit.article_id, title=hit.title, content=content
                    )
                )
        return articles

    async def _write_answer(
        self,
        request: ChatRequest,
        articles: list[AnswerContextArticle],
        hits: list[RetrievedArticle],
        result: ChatResult,
    ) -> None:
        if self._answerer is None:
            self._fallback(hits, result)
            return
        try:
            generated = await self._answerer.answer(
                request.question, self._query.recent_history(request.history), articles
            )
        except Exception as exc:
            # Same resilience rule as the rewrite: an LLM failure never fails the request.
            logger.warning(
                "answer generation failed, quoting the best article",
                extra={"exception_type": type(exc).__name__},
            )
            self._fallback(hits, result)
            return

        result.kind = generated.kind
        if generated.kind is AnswerKind.OFF_TOPIC:
            result.answer = OFF_TOPIC_MESSAGE
        elif generated.kind is AnswerKind.NOT_FOUND:
            result.answer = NOT_FOUND_MESSAGE
        elif generated.kind is AnswerKind.SMALL_TALK:
            result.answer = generated.answer
        else:
            result.answer = generated.answer
            titles = {a.article_id: a.title for a in articles}
            # Only sources that were really provided: the model cannot invent an article ID.
            used = [i for i in dict.fromkeys(generated.source_article_ids) if i in titles]
            result.sources = [ChatSource(article_id=i, title=titles[i]) for i in used]

    @staticmethod
    def _fallback(hits: list[RetrievedArticle], result: ChatResult) -> None:
        result.answer_fallback = True
        if not hits:
            result.kind, result.answer = AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE
            return
        best = hits[0]
        result.kind = AnswerKind.ANSWER
        result.answer = f"{FALLBACK_INTRO}\n\n{best.title}\n{best.snippet}"
        result.sources = [ChatSource(article_id=best.article_id, title=best.title)]
