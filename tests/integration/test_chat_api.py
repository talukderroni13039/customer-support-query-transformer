from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core import lifespan as lifespan_module
from src.core.exceptions import LLMOutputInvalidError, LLMTimeoutError
from src.core.lifespan import AppResources
from src.domains.support_search.schemas import AnswerKind
from src.llm.cached_answerer import CachedAnswerGenerator
from tests.integration.test_articles_api import err, ok

ARTICLES = "/api/v1/support/articles"
CHAT = "/api/v1/support/chat"

KB = [
    ("Returns for Outerwear", "Jackets can be returned within 60 days of delivery.", "returns"),
    ("Shipping to Canada", "Standard shipping to Canada takes 6 to 10 business days.", "shipping"),
]


@pytest.fixture
def seeded_client(client: TestClient) -> TestClient:
    for title, content, category in KB:
        body = {"title": title, "content": content, "category": category}
        ok(client.post(ARTICLES, json=body), 201)
    return client


def chat(client: TestClient, **body: Any) -> Any:
    return client.post(CHAT, json=body)


def test_chat_returns_one_answer_with_citations_in_the_envelope(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.answer_generator.answer_text = "Shipping to Canada takes 6 to 10 business days."  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="How long does shipping to Canada take?"))
    assert data["kind"] == "answer"
    assert data["answer"] == "Shipping to Canada takes 6 to 10 business days."
    assert [s["title"] for s in data["sources"]] == ["Shipping to Canada"]
    assert data["citations"][0]["title"] == "Shipping to Canada"
    assert "6 to 10 business days" in data["citations"][0]["quote"]
    assert data["answer_fallback"] is False
    assert data["grounding_failed"] is False
    assert data["guardrail_actions"] == []
    assert "results" not in data  # one answer, not a ranked list


def test_chat_follow_up_with_history(seeded_client: TestClient, resources: AppResources) -> None:
    resources.rewriter.standalone_query = "Can jackets be returned within 60 days of delivery"  # type: ignore[union-attr]
    history = ["Do you sell jackets?", "Yes, the Trailblazer Pro."]
    data = ok(chat(seeded_client, question="How long do I have to return it?", history=history))
    assert data["search_query"] == "Can jackets be returned within 60 days of delivery"
    assert data["sources"][0]["title"] == "Returns for Outerwear"
    assert resources.answer_generator.calls[0][1] == history  # type: ignore[union-attr]


def test_off_topic_question_reaches_the_model_with_no_chunks(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.answer_generator.kind = AnswerKind.OFF_TOPIC  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="zzzz qqqq xxxx"))
    assert data["kind"] == "off_topic"
    assert "only help with Northwind Gear support" in data["answer"]
    assert data["sources"] == []
    assert resources.answer_generator.calls[0][2] == []  # type: ignore[union-attr]


def test_not_found_message(seeded_client: TestClient, resources: AppResources) -> None:
    resources.answer_generator.kind = AnswerKind.NOT_FOUND  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="Can jackets be returned in Brazil?"))
    assert data["kind"] == "not_found"
    assert "don't have that information" in data["answer"]


def test_chat_small_talk(seeded_client: TestClient, resources: AppResources) -> None:
    resources.rewriter.needs_retrieval = False  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="Thanks!", history=["a question", "an answer"]))
    assert data["kind"] == "small_talk"
    assert data["answer"]
    assert resources.answer_generator.calls == []  # type: ignore[union-attr]


def test_ungrounded_answer_is_withheld(seeded_client: TestClient, resources: AppResources) -> None:
    resources.answer_generator.error = LLMOutputInvalidError()  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="How long does shipping to Canada take?"))
    assert data["grounding_failed"] is True
    assert data["kind"] == "not_found"
    assert data["citations"] == []


def test_output_guardrail_is_reported(seeded_client: TestClient, resources: AppResources) -> None:
    resources.answer_generator.answer_text = "Call +1 (555) 010-9999 about shipping."  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="How long does shipping to Canada take?"))
    assert "555" not in data["answer"]
    assert data["guardrail_actions"] == ["masked_phone_number"]


def test_chat_never_fails_when_the_llm_is_down(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.answer_generator.error = LLMTimeoutError()  # type: ignore[union-attr]
    data = ok(chat(seeded_client, question="How long does shipping to Canada take?"))
    assert data["answer_fallback"] is True
    assert "6 to 10 business days" in data["answer"]

    resources.answer_generator = None
    resources.reranker = None
    data = ok(chat(seeded_client, question="How long does shipping to Canada take?"))
    assert data["answer_fallback"] is True


def test_chat_threshold_is_configurable(
    seeded_client: TestClient, resources: AppResources, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MIN_RERANK_SCORE", "0.99")
    monkeypatch.setenv("CHAT_CONTEXT_CHUNKS", "1")
    resources.answer_generator.kind = AnswerKind.NOT_FOUND  # type: ignore[union-attr]
    ok(chat(seeded_client, question="shipping a kayak tomorrow"))
    assert resources.answer_generator.calls[0][2] == []  # type: ignore[union-attr]


def test_chat_validation_and_empty_knowledge_base(client: TestClient) -> None:
    err(chat(client, question="anything"), 409, "KB_EMPTY")
    for body in (
        {},
        {"question": ""},
        {"question": "q" * 2001},
        {"question": "q", "history": ["h"] * 101},
        {"question": "q", "history": [{"role": "user", "content": "x"}]},
        {"question": "q", "top_k": 3},
        {"question": "q", "rerank": False},
    ):
        err(client.post(CHAT, json=body), 422, "VALIDATION_ERROR")


def test_build_answer_generator_provider_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("REWRITE_PROVIDER", "none")
    assert lifespan_module.build_answer_generator() is None
    monkeypatch.setenv("REWRITE_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert lifespan_module.build_answer_generator() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ANSWER_CACHE_PATH", str(tmp_path / "answers.json"))
    monkeypatch.setenv("LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    assert isinstance(lifespan_module.build_answer_generator(), CachedAnswerGenerator)
