"""AnthropicAnswerGenerator and CachedAnswerGenerator against a mocked HTTP transport."""

import json
from pathlib import Path

import httpx
import pytest

from src.core.exceptions import LLMBudgetExhaustedError, LLMError, LLMOutputInvalidError
from src.domains.support_search.prompts.answer_prompt import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_message,
)
from src.domains.support_search.schemas import AnswerKind, ContextChunk
from src.llm.anthropic_answerer import AnthropicAnswerGenerator
from src.llm.cached_answerer import CachedAnswerGenerator
from src.llm.call_budget import LlmCallBudget
from tests.fakes import FakeAnswerGenerator
from tests.unit.test_anthropic_rewriter import MockAnthropic

QUESTION = "How long do I have to return it?"
HISTORY = ["Do you sell jackets?", "Yes, the Trailblazer Pro (TB-PRO-22)."]
CHUNKS = [
    ContextChunk(
        label="C1",
        chunk_id="KB-012::0000",
        article_id="KB-012",
        title="Returns for Outerwear",
        heading="Return window",
        text="## Return window\nJackets can be returned within 60 days of delivery.",
    ),
    ContextChunk(
        label="C2",
        chunk_id="KB-031::0000",
        article_id="KB-031",
        title="Shipping to Canada",
        text="Standard shipping to Canada takes 6 to 10 business days.",
    ),
]
GOOD = {
    "kind": "answer",
    "answer": "You have 60 days from delivery to return the jacket.",
    "citations": [{"chunk_id": "C1", "quote": "can be returned within 60 days of delivery"}],
}
INVENTED = {
    "kind": "answer",
    "answer": "You have 90 days from delivery to return the jacket.",
    "citations": [{"chunk_id": "C1", "quote": "can be returned within 90 days of delivery"}],
}


def make_generator(
    mock: MockAnthropic, tmp_path: Path, limit: int = 50
) -> tuple[AnthropicAnswerGenerator, LlmCallBudget]:
    budget = LlmCallBudget(tmp_path / "llm_call_budget.json", limit=limit)
    generator = AnthropicAnswerGenerator(
        api_key="test-key-not-real", budget=budget, transport=httpx.MockTransport(mock)
    )
    return generator, budget


async def test_grounded_answer_is_returned_and_counts_one_call(tmp_path: Path) -> None:
    generator, budget = make_generator(MockAnthropic(GOOD), tmp_path)
    result = await generator.answer(QUESTION, HISTORY, CHUNKS)
    assert result.kind is AnswerKind.ANSWER
    assert result.answer == GOOD["answer"]
    assert result.citations[0].chunk_id == "C1"
    assert result.from_cache is False
    assert budget.used() == 1
    await generator.aclose()


async def test_request_contains_labelled_chunks_history_and_question(tmp_path: Path) -> None:
    mock = MockAnthropic({"kind": "off_topic"})
    generator, _ = make_generator(mock, tmp_path)
    result = await generator.answer("What is the capital of France?", HISTORY, CHUNKS)
    assert result.kind is AnswerKind.OFF_TOPIC

    body = mock.requests[0]
    assert body["model"] == "claude-haiku-5-5"
    for sampling_param in ("temperature", "top_p", "top_k"):  # rejected by this model family
        assert sampling_param not in body
    user_text = str(body["messages"][0]["content"])
    assert "[C1] Returns for Outerwear / Return window" in user_text
    assert "Jackets can be returned within 60 days of delivery." in user_text
    assert "## Return window" not in user_text  # the heading line is not repeated in the body
    assert "[C2] Shipping to Canada" in user_text
    assert "- Do you sell jackets?" in user_text
    assert "What is the capital of France?" in user_text
    properties = body["tools"][0]["input_schema"]["properties"]
    assert set(properties) == {"kind", "answer", "citations"}


async def test_ungrounded_answer_is_sent_back_once_and_the_correction_is_accepted(
    tmp_path: Path,
) -> None:
    mock = MockAnthropic(INVENTED, GOOD)
    generator, budget = make_generator(mock, tmp_path)
    result = await generator.answer(QUESTION, HISTORY, CHUNKS)
    assert "60 days" in result.answer
    assert budget.used() == 2
    # The correction request tells Claude what failed.
    retry_text = json.dumps(mock.requests[1]["messages"])
    assert "not grounded" in retry_text
    assert "90" in retry_text


async def test_answer_that_stays_ungrounded_raises_output_invalid(tmp_path: Path) -> None:
    generator, budget = make_generator(MockAnthropic(INVENTED), tmp_path)
    with pytest.raises(LLMOutputInvalidError):
        await generator.answer(QUESTION, HISTORY, CHUNKS)
    assert budget.used() == 2  # first attempt + exactly one correction attempt


@pytest.mark.parametrize(
    "bad",
    [
        {
            **GOOD,
            "citations": [{"chunk_id": "C7", "quote": "anything at all here"}],
        },  # no such chunk
        {**GOOD, "citations": []},  # an answer needs a citation
        {
            **GOOD,
            "answer": "Jackets: 60 days. Shipping is 12 to 14 days.",
            "citations": GOOD["citations"],
        },
    ],
)
async def test_other_grounding_failures_are_rejected(tmp_path: Path, bad: dict) -> None:  # type: ignore[type-arg]
    generator, _ = make_generator(MockAnthropic(bad), tmp_path)
    with pytest.raises(LLMOutputInvalidError):
        await generator.answer(QUESTION, HISTORY, CHUNKS)


async def test_answer_with_no_chunks_cannot_be_an_answer(tmp_path: Path) -> None:
    generator, _ = make_generator(MockAnthropic(GOOD), tmp_path)
    with pytest.raises(LLMOutputInvalidError):
        await generator.answer(QUESTION, HISTORY, [])  # nothing to cite → never an answer
    generator, _ = make_generator(MockAnthropic({"kind": "not_found"}), tmp_path / "b")
    assert (await generator.answer(QUESTION, HISTORY, [])).kind is AnswerKind.NOT_FOUND


async def test_numbers_from_the_conversation_are_allowed(tmp_path: Path) -> None:
    answer = {
        "kind": "answer",
        "answer": "Your TB-PRO-22 jacket can be returned within 60 days of delivery.",
        "citations": GOOD["citations"],
    }
    generator, _ = make_generator(MockAnthropic(answer), tmp_path)
    result = await generator.answer(QUESTION, HISTORY, CHUNKS)  # "22" comes from the history
    assert "TB-PRO-22" in result.answer


async def test_api_errors_and_exhausted_budget_raise_llm_errors(tmp_path: Path) -> None:
    generator, budget = make_generator(MockAnthropic(400), tmp_path)
    with pytest.raises(LLMError, match="HTTP 400") as excinfo:
        await generator.answer(QUESTION, HISTORY, CHUNKS)
    assert not isinstance(excinfo.value, LLMOutputInvalidError)
    assert budget.used() == 1

    generator, budget = make_generator(MockAnthropic({"kind": "off_topic"}), tmp_path / "b", 1)
    await generator.answer(QUESTION, HISTORY, CHUNKS)
    with pytest.raises(LLMBudgetExhaustedError):
        await generator.answer(QUESTION, HISTORY, CHUNKS)


def test_prompt_rules_and_user_message() -> None:
    prompt = ANSWER_SYSTEM_PROMPT.lower()
    for rule in (
        "must be written in a chunk",
        "off_topic",
        "not_found",
        "do not guess",
        "word for word",
    ):
        assert rule in prompt
    empty = build_answer_user_message("hi", [], [])
    assert "(no relevant chunks were found)" in empty
    assert "(no earlier messages)" in empty


async def test_cached_answers_are_reused_and_keyed_by_chunk_text(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "answers.json"
    inner = FakeAnswerGenerator(answer="Within 60 days.")
    cached = CachedAnswerGenerator(inner, path, "model:v2")

    first = await cached.answer(QUESTION, HISTORY, CHUNKS)
    second = await cached.answer(QUESTION, HISTORY, CHUNKS)
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.answer == "Within 60 days."
    assert second.citations[0].chunk_id == "C1"
    assert len(inner.calls) == 1

    raw = path.read_text(encoding="utf-8")
    assert QUESTION not in raw
    assert HISTORY[0] not in raw

    edited = [CHUNKS[0].model_copy(update={"text": "Jackets: 90 days now."}), CHUNKS[1]]
    await cached.answer(QUESTION, HISTORY, edited)  # chunk text changed → not reused
    await cached.answer(QUESTION + "?", HISTORY, CHUNKS)
    await cached.answer(QUESTION, [], CHUNKS)
    await cached.answer(QUESTION, HISTORY, CHUNKS[:1])
    assert len(inner.calls) == 5

    reopened_inner = FakeAnswerGenerator(answer="unused")
    reopened = CachedAnswerGenerator(reopened_inner, path, "model:v2")
    assert (await reopened.answer(QUESTION, HISTORY, CHUNKS)).from_cache is True
    assert reopened_inner.calls == []
    other_version = CachedAnswerGenerator(reopened_inner, path, "model:v3")
    assert (await other_version.answer(QUESTION, HISTORY, CHUNKS)).from_cache is False


async def test_cached_answerer_does_not_cache_failures_and_survives_bad_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answers.json"
    path.write_text("{ not json", encoding="utf-8")
    inner = FakeAnswerGenerator(error=LLMOutputInvalidError())
    cached = CachedAnswerGenerator(inner, path, "model:v2")
    with pytest.raises(LLMOutputInvalidError):
        await cached.answer(QUESTION, HISTORY, CHUNKS)
    assert path.read_text(encoding="utf-8") == "{ not json"  # a failure writes nothing
    inner.error = None
    assert (await cached.answer(QUESTION, HISTORY, CHUNKS)).from_cache is False
    path.write_text("[1]", encoding="utf-8")
    assert (
        await CachedAnswerGenerator(inner, path, "model:v2").answer(QUESTION, HISTORY, CHUNKS)
    ).from_cache is False
