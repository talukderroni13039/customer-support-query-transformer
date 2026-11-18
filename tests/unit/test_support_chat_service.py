import pytest

from src.core.exceptions import KnowledgeBaseEmptyError, LLMBudgetExhaustedError, LLMTimeoutError
from src.core.lifespan import AppResources
from src.domains.support_search.schemas import AnswerKind, ArticleCreate, ChatRequest
from src.domains.support_search.services.article_service import ArticleService
from src.domains.support_search.services.query_service import QueryService
from src.domains.support_search.services.retrieval_service import RetrievalService
from src.domains.support_search.services.support_chat_service import (
    FALLBACK_INTRO,
    NOT_FOUND_MESSAGE,
    OFF_TOPIC_MESSAGE,
    SMALL_TALK_MESSAGE,
    SupportChatService,
)
from src.domains.support_search.services.support_search_service import SupportSearchService
from tests.fakes import FakeAnswerGenerator, FakeRewriter

RETURNS = (
    "## Return window\nJackets can be returned within 60 days of delivery.\n\n"
    "## Condition\nJackets must be unworn with tags attached.\n"
)
ARTICLES = {
    "KB-012": ("Returns for Outerwear", RETURNS),
    "KB-031": ("Shipping to Canada", "Standard shipping to Canada takes 6 to 10 business days."),
    "KB-061": ("Gift Cards", "Gift cards and store credit never expire and have no fees."),
}
HISTORY = ["Do you sell jackets?", "Yes, the Trailblazer Pro jacket."]


@pytest.fixture
async def seeded(resources: AppResources) -> AppResources:
    articles = ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
    )
    for article_id, (title, content) in ARTICLES.items():
        await articles.upsert(article_id, ArticleCreate(title=title, content=content))
    return resources


def make_service(
    resources: AppResources,
    answerer: FakeAnswerGenerator | None,
    rewriter: FakeRewriter | None = None,
) -> SupportChatService:
    query_service = QueryService(rewriter)
    search = SupportSearchService(
        query_service=query_service,
        retrieval_service=RetrievalService(
            embedder=resources.embedder,
            vector_store=resources.vector_store,
            keyword_index=resources.keyword_index,
            reranker=resources.reranker,
        ),
        keyword_index=resources.keyword_index,
    )
    return SupportChatService(
        search_service=search,
        query_service=query_service,
        vector_store=resources.vector_store,
        answer_generator=answerer,
        rerank_available=resources.reranker is not None,
    )


async def test_answer_is_written_from_full_articles_with_sources(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(answer="You have 60 days to return a jacket.")
    result = await make_service(seeded, answerer).chat(
        ChatRequest(question="How long can jackets be returned after delivery?")
    )
    assert result.kind is AnswerKind.ANSWER
    assert result.answer == "You have 60 days to return a jacket."
    assert [s.article_id for s in result.sources] == ["KB-012"]
    assert result.sources[0].title == "Returns for Outerwear"
    assert result.answer_fallback is False

    question, history, articles = answerer.calls[0]
    assert question == "How long can jackets be returned after delivery?"
    assert history == []
    assert articles[0].article_id == "KB-012"
    assert articles[0].content == RETURNS  # the whole article, not just the matching snippet
    assert len(articles) <= 3


async def test_follow_up_uses_the_rewritten_query_and_passes_history(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    rewriter = FakeRewriter("Can jackets be returned within 60 days of delivery")
    result = await make_service(seeded, answerer, rewriter).chat(
        ChatRequest(question="How long do I have to return it?", history=HISTORY)
    )
    assert result.search_query == "Can jackets be returned within 60 days of delivery"
    assert result.kind is AnswerKind.ANSWER
    question, history, articles = answerer.calls[0]
    assert question == "How long do I have to return it?"  # the customer's own words
    assert history == HISTORY
    assert articles[0].article_id == "KB-012"


async def test_only_the_last_twelve_history_items_reach_the_answer_generator(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator()
    history = [f"message {i}" for i in range(40)]
    await make_service(seeded, answerer, FakeRewriter("gift cards expire")).chat(
        ChatRequest(question="Do they expire?", history=history)
    )
    assert answerer.calls[0][1] == history[-12:]


async def test_off_topic_gets_the_fixed_message_and_no_sources(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(kind=AnswerKind.OFF_TOPIC)
    result = await make_service(seeded, answerer).chat(
        ChatRequest(question="What is the capital of France?")
    )
    assert result.kind is AnswerKind.OFF_TOPIC
    assert result.answer == OFF_TOPIC_MESSAGE
    assert result.sources == []


async def test_support_question_without_an_answer_gets_not_found(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(kind=AnswerKind.NOT_FOUND)
    result = await make_service(seeded, answerer).chat(
        ChatRequest(question="Do you ship to Brazil?")
    )
    assert result.kind is AnswerKind.NOT_FOUND
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


async def test_small_talk_seen_by_the_rewriter_skips_retrieval_and_the_answer_call(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator()
    result = await make_service(seeded, answerer, FakeRewriter(needs_retrieval=False)).chat(
        ChatRequest(question="Thanks, that's all!", history=HISTORY)
    )
    assert result.kind is AnswerKind.SMALL_TALK
    assert result.answer == SMALL_TALK_MESSAGE
    assert answerer.calls == []


async def test_small_talk_without_history_is_answered_by_the_generator(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator(kind=AnswerKind.SMALL_TALK, answer="Hello! How can I help?")
    result = await make_service(seeded, answerer).chat(ChatRequest(question="hi"))
    assert result.kind is AnswerKind.SMALL_TALK
    assert result.answer == "Hello! How can I help?"
    assert result.sources == []


async def test_invented_or_duplicate_source_ids_are_dropped(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(source_article_ids=["KB-012", "KB-999", "KB-012"])
    result = await make_service(seeded, answerer).chat(
        ChatRequest(question="How long can jackets be returned after delivery?")
    )
    assert [s.article_id for s in result.sources] == ["KB-012"]


@pytest.mark.parametrize("error", [LLMTimeoutError(), LLMBudgetExhaustedError(), RuntimeError("x")])
async def test_answer_failure_falls_back_to_the_best_article(
    seeded: AppResources, error: Exception
) -> None:
    result = await make_service(seeded, FakeAnswerGenerator(error=error)).chat(
        ChatRequest(question="How long can jackets be returned after delivery?")
    )
    assert result.answer_fallback is True
    assert result.kind is AnswerKind.ANSWER
    assert result.answer.startswith(FALLBACK_INTRO)
    assert "Returns for Outerwear" in result.answer
    assert "60 days" in result.answer
    assert [s.article_id for s in result.sources] == ["KB-012"]


async def test_no_answer_generator_configured_falls_back(seeded: AppResources) -> None:
    result = await make_service(seeded, None).chat(
        ChatRequest(question="How long does shipping to Canada take?")
    )
    assert result.answer_fallback is True
    assert "Shipping to Canada" in result.answer


async def test_rewrite_failure_is_reported_and_chat_still_answers(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    result = await make_service(seeded, answerer, FakeRewriter(error=LLMTimeoutError())).chat(
        ChatRequest(question="shipping to Canada", history=HISTORY)
    )
    assert result.rewrite_fallback is True
    assert result.kind is AnswerKind.ANSWER
    assert len(answerer.calls) == 1


async def test_chat_works_without_a_reranker(seeded: AppResources) -> None:
    seeded.reranker = None
    result = await make_service(seeded, FakeAnswerGenerator()).chat(
        ChatRequest(question="How long does shipping to Canada take?")
    )
    assert result.kind is AnswerKind.ANSWER
    assert result.sources[0].article_id == "KB-031"


async def test_empty_knowledge_base_raises_before_any_llm_call(resources: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    rewriter = FakeRewriter()
    with pytest.raises(KnowledgeBaseEmptyError):
        await make_service(resources, answerer, rewriter).chat(
            ChatRequest(question="anything", history=HISTORY)
        )
    assert answerer.calls == []
    assert rewriter.calls == []
