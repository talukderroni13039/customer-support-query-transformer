import pytest

from src.core.exceptions import (
    KnowledgeBaseEmptyError,
    LLMBudgetExhaustedError,
    LLMOutputInvalidError,
    LLMTimeoutError,
)
from src.core.lifespan import AppResources
from src.domains.support_search.schemas import AnswerKind, ArticleCreate, ChatRequest, Citation
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
RETURN_QUESTION = "Can jackets be returned within 60 days of delivery?"


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
    **kwargs: float,
) -> SupportChatService:
    return SupportChatService(
        query_service=QueryService(rewriter),
        retrieval_service=RetrievalService(
            embedder=resources.embedder,
            vector_store=resources.vector_store,
            keyword_index=resources.keyword_index,
            reranker=resources.reranker,
        ),
        keyword_index=resources.keyword_index,
        answer_generator=answerer,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_answer_comes_from_labelled_chunks_with_citations(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(answer="You have 60 days to return a jacket.")
    result = await make_service(seeded, answerer).chat(ChatRequest(question=RETURN_QUESTION))

    assert result.kind is AnswerKind.ANSWER
    assert result.answer == "You have 60 days to return a jacket."
    assert [s.article_id for s in result.sources] == ["KB-012"]
    assert result.citations[0].title == "Returns for Outerwear"
    assert result.citations[0].heading == "Return window"
    assert "60 days" in result.citations[0].quote
    assert (result.answer_fallback, result.grounding_failed) == (False, False)
    assert result.guardrail_actions == []

    question, history, chunks = answerer.calls[0]
    assert question == RETURN_QUESTION
    assert history == []
    assert [c.label for c in chunks] == [f"C{i}" for i in range(1, len(chunks) + 1)]
    assert chunks[0].article_id == "KB-012"
    assert chunks[0].source == "Returns for Outerwear > Return window"
    assert len(chunks) <= 5


async def test_threshold_gate_drops_weak_chunks(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    await make_service(seeded, answerer, min_rerank_score=0.3).chat(
        ChatRequest(question=RETURN_QUESTION)
    )
    chunks = answerer.calls[0][2]
    assert chunks
    assert all(c.score >= 0.3 for c in chunks)
    assert {c.article_id for c in chunks} == {"KB-012"}  # shipping and gift cards did not pass

    lenient = FakeAnswerGenerator()
    await make_service(seeded, lenient, min_rerank_score=0.0).chat(
        ChatRequest(question=RETURN_QUESTION)
    )
    assert len(lenient.calls[0][2]) > len(chunks)


async def test_when_nothing_passes_the_gate_the_model_gets_no_chunks(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(kind=AnswerKind.OFF_TOPIC)
    result = await make_service(seeded, answerer).chat(ChatRequest(question="zzzz qqqq xxxx"))
    assert answerer.calls[0][2] == []  # it can classify the message, but has nothing to answer from
    assert result.kind is AnswerKind.OFF_TOPIC
    assert result.answer == OFF_TOPIC_MESSAGE
    assert result.citations == []


async def test_context_is_capped_at_max_chunks(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    await make_service(seeded, answerer, min_rerank_score=0.0, max_context_chunks=2).chat(
        ChatRequest(question=RETURN_QUESTION)
    )
    assert len(answerer.calls[0][2]) == 2


async def test_follow_up_uses_the_rewritten_query_and_passes_history(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    rewriter = FakeRewriter("Can jackets be returned within 60 days of delivery")
    result = await make_service(seeded, answerer, rewriter).chat(
        ChatRequest(question="How long do I have to return it?", history=HISTORY)
    )
    assert result.search_query == "Can jackets be returned within 60 days of delivery"
    question, history, chunks = answerer.calls[0]
    assert question == "How long do I have to return it?"  # the customer's own words
    assert history == HISTORY
    assert chunks[0].article_id == "KB-012"


async def test_only_the_last_twelve_history_items_reach_the_answer_generator(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator()
    history = [f"message {i}" for i in range(40)]
    await make_service(seeded, answerer, FakeRewriter("gift cards expire")).chat(
        ChatRequest(question="Do they expire?", history=history)
    )
    assert answerer.calls[0][1] == history[-12:]


async def test_not_found_and_off_topic_use_fixed_messages(seeded: AppResources) -> None:
    result = await make_service(seeded, FakeAnswerGenerator(kind=AnswerKind.NOT_FOUND)).chat(
        ChatRequest(question="Can jackets be returned to a store in Brazil?")
    )
    assert (result.kind, result.answer) == (AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE)
    assert result.sources == []
    result = await make_service(seeded, FakeAnswerGenerator(kind=AnswerKind.OFF_TOPIC)).chat(
        ChatRequest(question="What is the capital of France?")
    )
    assert (result.kind, result.answer) == (AnswerKind.OFF_TOPIC, OFF_TOPIC_MESSAGE)


async def test_small_talk_seen_by_the_rewriter_skips_retrieval_and_the_answer_call(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator()
    result = await make_service(seeded, answerer, FakeRewriter(needs_retrieval=False)).chat(
        ChatRequest(question="Thanks, that's all!", history=HISTORY)
    )
    assert (result.kind, result.answer) == (AnswerKind.SMALL_TALK, SMALL_TALK_MESSAGE)
    assert answerer.calls == []


async def test_small_talk_without_history_is_answered_by_the_generator(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator(kind=AnswerKind.SMALL_TALK, answer="Hello! How can I help?")
    result = await make_service(seeded, answerer).chat(ChatRequest(question="hi"))
    assert (result.kind, result.answer) == (AnswerKind.SMALL_TALK, "Hello! How can I help?")


async def test_grounding_failure_withholds_the_answer(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(error=LLMOutputInvalidError())
    result = await make_service(seeded, answerer).chat(ChatRequest(question=RETURN_QUESTION))
    assert result.grounding_failed is True
    assert result.answer_fallback is False
    assert (result.kind, result.answer) == (AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE)
    assert result.citations == []


async def test_citations_of_unknown_chunks_are_never_shown(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator(citations=[Citation(chunk_id="C9", quote="made up")])
    result = await make_service(seeded, answerer).chat(ChatRequest(question=RETURN_QUESTION))
    assert result.grounding_failed is True
    assert result.kind is AnswerKind.NOT_FOUND


async def test_output_guardrail_masks_details_not_in_the_cited_chunks(
    seeded: AppResources,
) -> None:
    answerer = FakeAnswerGenerator(answer="Email bob@example.com within 60 days.")
    result = await make_service(seeded, answerer).chat(ChatRequest(question=RETURN_QUESTION))
    assert "bob@example.com" not in result.answer
    assert result.guardrail_actions == ["masked_email"]
    assert result.kind is AnswerKind.ANSWER


@pytest.mark.parametrize("error", [LLMTimeoutError(), LLMBudgetExhaustedError(), RuntimeError("x")])
async def test_llm_outage_falls_back_to_quoting_the_best_chunk(
    seeded: AppResources, error: Exception
) -> None:
    result = await make_service(seeded, FakeAnswerGenerator(error=error)).chat(
        ChatRequest(question=RETURN_QUESTION)
    )
    assert result.answer_fallback is True
    assert result.grounding_failed is False
    assert result.kind is AnswerKind.ANSWER
    assert result.answer.startswith(FALLBACK_INTRO)
    assert "60 days" in result.answer
    assert [s.article_id for s in result.sources] == ["KB-012"]


async def test_outage_with_nothing_relevant_says_not_found(seeded: AppResources) -> None:
    result = await make_service(seeded, None).chat(ChatRequest(question="zzzz qqqq xxxx"))
    assert result.answer_fallback is True
    assert (result.kind, result.answer) == (AnswerKind.NOT_FOUND, NOT_FOUND_MESSAGE)


async def test_rewrite_failure_is_reported_and_chat_still_answers(seeded: AppResources) -> None:
    answerer = FakeAnswerGenerator()
    result = await make_service(seeded, answerer, FakeRewriter(error=LLMTimeoutError())).chat(
        ChatRequest(question="shipping to Canada", history=HISTORY)
    )
    assert result.rewrite_fallback is True
    assert result.kind is AnswerKind.ANSWER
    assert len(answerer.calls) == 1


async def test_without_a_reranker_the_top_chunks_pass_ungated(seeded: AppResources) -> None:
    seeded.reranker = None
    answerer = FakeAnswerGenerator()
    result = await make_service(seeded, answerer, min_rerank_score=0.99).chat(
        ChatRequest(question="How long does shipping to Canada take?")
    )
    assert result.kind is AnswerKind.ANSWER
    assert answerer.calls[0][2][0].article_id == "KB-031"


async def test_empty_knowledge_base_raises_before_any_llm_call(resources: AppResources) -> None:
    answerer, rewriter = FakeAnswerGenerator(), FakeRewriter()
    with pytest.raises(KnowledgeBaseEmptyError):
        await make_service(resources, answerer, rewriter).chat(
            ChatRequest(question="anything", history=HISTORY)
        )
    assert answerer.calls == []
    assert rewriter.calls == []
