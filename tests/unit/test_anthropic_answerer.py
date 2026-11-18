"""AnthropicAnswerGenerator and CachedAnswerGenerator against a mocked HTTP transport."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from src.core.exceptions import LLMBudgetExhaustedError, LLMError
from src.domains.support_search.prompts.answer_prompt import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_message,
)
from src.domains.support_search.schemas import AnswerContextArticle, AnswerKind, GeneratedAnswer
from src.llm.anthropic_answerer import AnthropicAnswerGenerator
from src.llm.cached_answerer import CachedAnswerGenerator
from src.llm.call_budget import LlmCallBudget
from tests.fakes import FakeAnswerGenerator
from tests.unit.test_anthropic_rewriter import MockAnthropic

QUESTION = "How long do I have to return it?"
HISTORY = ["Do you sell jackets?", "Yes, the Trailblazer Pro (TB-PRO-22)."]
ARTICLES = [
    AnswerContextArticle(
        article_id="KB-012",
        title="Returns for Outerwear",
        content="## Return window\nJackets can be returned within 60 days of delivery.",
    )
]


def make_generator(
    mock: MockAnthropic, tmp_path: Path, limit: int = 50
) -> tuple[AnthropicAnswerGenerator, LlmCallBudget]:
    budget = LlmCallBudget(tmp_path / "llm_call_budget.json", limit=limit)
    generator = AnthropicAnswerGenerator(
        api_key="test-key-not-real",
        budget=budget,
        max_retries=1,
        transport=httpx.MockTransport(mock),
    )
    return generator, budget


async def test_answer_returns_typed_result_and_counts_one_call(tmp_path: Path) -> None:
    mock = MockAnthropic(
        {
            "kind": "answer",
            "answer": "You have 60 days from delivery to return the jacket.",
            "source_article_ids": ["KB-012"],
        }
    )
    generator, budget = make_generator(mock, tmp_path)
    result = await generator.answer(QUESTION, HISTORY, ARTICLES)
    assert result.kind is AnswerKind.ANSWER
    assert result.answer == "You have 60 days from delivery to return the jacket."
    assert result.source_article_ids == ["KB-012"]
    assert result.from_cache is False
    assert budget.used() == 1
    await generator.aclose()


async def test_request_contains_articles_history_and_question(tmp_path: Path) -> None:
    mock = MockAnthropic({"kind": "off_topic"})
    generator, _ = make_generator(mock, tmp_path)
    result = await generator.answer("What is the capital of France?", HISTORY, ARTICLES)
    assert result.kind is AnswerKind.OFF_TOPIC
    assert result.answer == ""

    body = mock.requests[0]
    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == 700
    assert "temperature" not in body
    assert "Northwind Gear" in str(body["system"])
    user_text = str(body["messages"][0]["content"])
    assert '<article id="KB-012" title="Returns for Outerwear">' in user_text
    assert "within 60 days of delivery" in user_text
    assert "1. Do you sell jackets?" in user_text
    assert "What is the capital of France?" in user_text
    properties = body["tools"][0]["input_schema"]["properties"]
    assert set(properties) == {"kind", "answer", "source_article_ids"}  # from_cache is hidden


async def test_empty_answer_is_retried_and_counted(tmp_path: Path) -> None:
    mock = MockAnthropic(
        {"kind": "answer", "answer": "   "},  # invalid: an answer needs text
        {"kind": "answer", "answer": "Within 60 days.", "source_article_ids": ["KB-012"]},
    )
    generator, budget = make_generator(mock, tmp_path)
    result = await generator.answer(QUESTION, HISTORY, ARTICLES)
    assert result.answer == "Within 60 days."
    assert budget.used() == 2


async def test_errors_and_exhausted_budget_raise_llm_errors(tmp_path: Path) -> None:
    generator, budget = make_generator(MockAnthropic(400), tmp_path)
    with pytest.raises(LLMError, match="HTTP 400"):
        await generator.answer(QUESTION, HISTORY, ARTICLES)
    assert budget.used() == 1

    generator, budget = make_generator(MockAnthropic({"kind": "off_topic"}), tmp_path / "b", 1)
    await generator.answer(QUESTION, HISTORY, ARTICLES)
    with pytest.raises(LLMBudgetExhaustedError):
        await generator.answer(QUESTION, HISTORY, ARTICLES)
    assert budget.used() == 1


def test_generated_answer_validation() -> None:
    assert GeneratedAnswer(kind="off_topic").answer == ""
    assert GeneratedAnswer(kind="not_found").source_article_ids == []
    assert GeneratedAnswer(kind="small_talk", answer="  Hello!  ").answer == "Hello!"
    for kind in ("answer", "small_talk"):
        with pytest.raises(ValidationError, match="answer must not be empty"):
            GeneratedAnswer(kind=kind, answer="")
    with pytest.raises(ValidationError):
        GeneratedAnswer(kind="answer", answer="x" * 4001)
    with pytest.raises(ValidationError):
        GeneratedAnswer(kind="made_up", answer="x")


def test_prompt_rules_and_user_message() -> None:
    prompt = ANSWER_SYSTEM_PROMPT.lower()
    for rule in ("only facts stated in the articles", "off_topic", "not_found", "do not guess"):
        assert rule in prompt
    empty = build_answer_user_message("hi", [], [])
    assert "(no articles were found)" in empty
    assert "(no earlier messages)" in empty


async def test_cached_answers_are_reused_and_keyed_by_article_text(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "answers.json"
    inner = FakeAnswerGenerator(answer="Within 60 days.")
    cached = CachedAnswerGenerator(inner, path, "model:v1")

    first = await cached.answer(QUESTION, HISTORY, ARTICLES)
    second = await cached.answer(QUESTION, HISTORY, ARTICLES)
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.answer == "Within 60 days."
    assert second.source_article_ids == ["KB-012"]
    assert len(inner.calls) == 1

    raw = path.read_text(encoding="utf-8")
    assert QUESTION not in raw
    assert HISTORY[0] not in raw
    assert next(iter(json.loads(raw).values()))["kind"] == "answer"

    edited = [ARTICLES[0].model_copy(update={"content": "Jackets: 90 days now."})]
    await cached.answer(QUESTION, HISTORY, edited)  # article text changed → not reused
    await cached.answer(QUESTION + "?", HISTORY, ARTICLES)
    await cached.answer(QUESTION, [], ARTICLES)
    assert len(inner.calls) == 4

    reopened_inner = FakeAnswerGenerator(answer="unused")
    reopened = CachedAnswerGenerator(reopened_inner, path, "model:v1")
    assert (await reopened.answer(QUESTION, HISTORY, ARTICLES)).from_cache is True
    assert reopened_inner.calls == []
    other_version = CachedAnswerGenerator(reopened_inner, path, "model:v2")
    assert (await other_version.answer(QUESTION, HISTORY, ARTICLES)).from_cache is False


async def test_cached_answerer_does_not_cache_failures_and_survives_bad_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answers.json"
    path.write_text("{ not json", encoding="utf-8")
    inner = FakeAnswerGenerator(error=RuntimeError("down"))
    cached = CachedAnswerGenerator(inner, path, "model:v1")
    with pytest.raises(RuntimeError):
        await cached.answer(QUESTION, HISTORY, ARTICLES)
    inner.error = None
    assert (await cached.answer(QUESTION, HISTORY, ARTICLES)).from_cache is False
    path.write_text("[1]", encoding="utf-8")
    assert (
        await CachedAnswerGenerator(inner, path, "model:v1").answer(QUESTION, HISTORY, ARTICLES)
    ).from_cache is False
