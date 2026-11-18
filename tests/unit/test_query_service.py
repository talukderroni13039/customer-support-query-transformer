import pytest

from src.core.exceptions import LLMBudgetExhaustedError, LLMError, LLMTimeoutError
from src.domains.support_search.schemas import QueryMode
from src.domains.support_search.services.query_service import QueryService
from tests.fakes import FakeRewriter

QUESTION = "How long do I have to return it?"
HISTORY = ["Do you sell the Trailblazer jacket?", "Yes, the Trailblazer Pro (TB-PRO-22)."]


async def test_baselines_never_call_the_rewriter() -> None:
    rewriter = FakeRewriter()
    service = QueryService(rewriter)

    question_only = await service.build_query(QUESTION, HISTORY, QueryMode.QUESTION_ONLY)
    assert question_only.search_query == QUESTION
    assert question_only.rewrite_source == "none"

    full_history = await service.build_query(QUESTION, HISTORY, QueryMode.FULL_HISTORY)
    assert full_history.search_query == "\n".join([*HISTORY, QUESTION])
    assert full_history.rewrite_source == "none"
    assert full_history.needs_retrieval is True
    assert full_history.rewrite_fallback is False

    assert rewriter.calls == []


@pytest.mark.parametrize("history", [[], ["", "   "]])
async def test_empty_history_skips_the_rewriter(history: list[str]) -> None:
    rewriter = FakeRewriter()
    plan = await QueryService(rewriter).build_query(QUESTION, history, QueryMode.REWRITE)
    assert plan.search_query == QUESTION
    assert plan.rewrite_source == "skipped"
    assert plan.rewrite_fallback is False
    assert rewriter.calls == []


async def test_rewrite_uses_the_standalone_query() -> None:
    rewriter = FakeRewriter("Return window for the Trailblazer Pro TB-PRO-22 jacket")
    plan = await QueryService(rewriter).build_query(QUESTION, HISTORY, QueryMode.REWRITE)
    assert plan.search_query == "Return window for the Trailblazer Pro TB-PRO-22 jacket"
    assert plan.rewrite_source == "llm"
    assert plan.needs_retrieval is True
    assert rewriter.calls == [(QUESTION, HISTORY)]


async def test_hundred_history_items_send_only_the_last_twelve() -> None:
    rewriter = FakeRewriter()
    history = [f"message {i}" for i in range(100)]
    service = QueryService(rewriter, max_history_messages=12)

    await service.build_query(QUESTION, history, QueryMode.REWRITE)
    assert rewriter.calls[0][1] == [f"message {i}" for i in range(88, 100)]

    full = await service.build_query(QUESTION, history, QueryMode.FULL_HISTORY)
    assert full.search_query.splitlines() == [*history[-12:], QUESTION]


async def test_small_talk_returns_no_retrieval() -> None:
    plan = await QueryService(FakeRewriter(needs_retrieval=False)).build_query(
        "Thanks, that's all!", HISTORY, QueryMode.REWRITE
    )
    assert plan.needs_retrieval is False
    assert plan.search_query == ""
    assert plan.rewrite_fallback is False


@pytest.mark.parametrize(
    "error", [LLMError(), LLMTimeoutError(), LLMBudgetExhaustedError(), RuntimeError("boom")]
)
async def test_rewriter_error_falls_back_to_the_question(error: Exception) -> None:
    plan = await QueryService(FakeRewriter(error=error)).build_query(
        QUESTION, HISTORY, QueryMode.REWRITE
    )
    assert plan.search_query == QUESTION
    assert plan.rewrite_fallback is True
    assert plan.rewrite_source == "fallback"
    assert plan.needs_retrieval is True


async def test_strict_mode_surfaces_llm_errors() -> None:
    with pytest.raises(LLMTimeoutError):
        await QueryService(FakeRewriter(error=LLMTimeoutError()), strict=True).build_query(
            QUESTION, HISTORY, QueryMode.REWRITE
        )
    with pytest.raises(LLMError):
        await QueryService(FakeRewriter(error=RuntimeError("boom")), strict=True).build_query(
            QUESTION, HISTORY, QueryMode.REWRITE
        )


async def test_rewrite_without_a_rewriter_is_flagged() -> None:
    plan = await QueryService(None).build_query(QUESTION, HISTORY, QueryMode.REWRITE)
    assert plan.search_query == QUESTION
    assert plan.rewrite_source == "disabled"
    assert plan.rewrite_fallback is True


async def test_cache_hits_are_reported_as_cache() -> None:
    class CachedFake(FakeRewriter):
        async def rewrite(self, question: str, history: list[str]):  # type: ignore[no-untyped-def]
            result = await super().rewrite(question, history)
            return result.model_copy(update={"from_cache": True})

    plan = await QueryService(CachedFake()).build_query(QUESTION, HISTORY, QueryMode.REWRITE)
    assert plan.rewrite_source == "cache"
