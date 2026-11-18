"""AnthropicRewriter against a mocked HTTP transport: the real Anthropic SDK and the real
Instructor run end to end, but no request ever leaves the process."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.core.exceptions import LLMBudgetExhaustedError, LLMError, LLMTimeoutError
from src.llm.anthropic_rewriter import AnthropicRewriter
from src.llm.call_budget import LlmCallBudget

QUESTION = "How long do I have to return it?"
HISTORY = ["Do you sell the Trailblazer jacket?", "Yes, the Trailblazer Pro (TB-PRO-22)."]


class MockAnthropic:
    """Replies to POST /v1/messages with scripted tool inputs, recording each request."""

    def __init__(self, *tool_inputs: Any) -> None:
        self.tool_inputs = list(tool_inputs)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        scripted = self.tool_inputs[min(len(self.requests), len(self.tool_inputs)) - 1]
        if isinstance(scripted, Exception):
            raise scripted
        if isinstance(scripted, int):
            return httpx.Response(
                scripted,
                json={"type": "error", "error": {"type": "api_error", "message": "scripted"}},
            )
        return httpx.Response(
            200,
            json={
                "id": f"msg_{len(self.requests)}",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{len(self.requests)}",
                        "name": body["tools"][0]["name"],
                        "input": scripted,
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )


def make_rewriter(
    mock: MockAnthropic, tmp_path: Path, *, limit: int = 50, max_retries: int = 1
) -> tuple[AnthropicRewriter, LlmCallBudget]:
    budget = LlmCallBudget(tmp_path / "llm_call_budget.json", limit=limit)
    rewriter = AnthropicRewriter(
        api_key="test-key-not-real",
        budget=budget,
        max_retries=max_retries,
        transport=httpx.MockTransport(mock),
    )
    return rewriter, budget


async def test_rewrite_returns_typed_result_and_counts_one_call(tmp_path: Path) -> None:
    mock = MockAnthropic(
        {"needs_retrieval": True, "standalone_query": "Return window for the TB-PRO-22 jacket"}
    )
    rewriter, budget = make_rewriter(mock, tmp_path)

    result = await rewriter.rewrite(QUESTION, HISTORY)

    assert result.needs_retrieval is True
    assert result.standalone_query == "Return window for the TB-PRO-22 jacket"
    assert result.from_cache is False
    assert budget.used() == 1
    assert len(mock.requests) == 1


async def test_request_shape_sent_to_anthropic(tmp_path: Path) -> None:
    mock = MockAnthropic({"needs_retrieval": True, "standalone_query": "q"})
    rewriter, _ = make_rewriter(mock, tmp_path)
    await rewriter.rewrite(QUESTION, HISTORY)

    body = mock.requests[0]
    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == 300
    assert "Northwind Gear" in str(body["system"])
    for sampling_param in ("temperature", "top_p", "top_k"):  # rejected by current models
        assert sampling_param not in body
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    user_text = str(body["messages"][0]["content"])
    assert "1. Do you sell the Trailblazer jacket?" in user_text
    assert "2. Yes, the Trailblazer Pro (TB-PRO-22)." in user_text
    assert QUESTION in user_text
    properties = body["tools"][0]["input_schema"]["properties"]
    assert set(properties) == {"needs_retrieval", "standalone_query"}  # from_cache is hidden


async def test_small_talk_result(tmp_path: Path) -> None:
    rewriter, _ = make_rewriter(MockAnthropic({"needs_retrieval": False}), tmp_path)
    result = await rewriter.rewrite("Thanks, that's all!", HISTORY)
    assert result.needs_retrieval is False
    assert result.standalone_query == ""


async def test_validation_retry_is_counted_as_a_separate_call(tmp_path: Path) -> None:
    mock = MockAnthropic(
        {"needs_retrieval": True, "standalone_query": ""},  # invalid → Instructor retries
        {"needs_retrieval": True, "standalone_query": "Return window TB-PRO-22"},
    )
    rewriter, budget = make_rewriter(mock, tmp_path, max_retries=1)
    result = await rewriter.rewrite(QUESTION, HISTORY)
    assert result.standalone_query == "Return window TB-PRO-22"
    assert len(mock.requests) == 2
    assert budget.used() == 2


async def test_retries_stop_at_max_retries_and_raise_llm_error(tmp_path: Path) -> None:
    mock = MockAnthropic({"needs_retrieval": True, "standalone_query": ""})
    rewriter, budget = make_rewriter(mock, tmp_path, max_retries=1)
    with pytest.raises(LLMError):
        await rewriter.rewrite(QUESTION, HISTORY)
    assert budget.used() == 2  # first attempt + exactly one retry


async def test_budget_exhausted_refuses_before_any_request(tmp_path: Path) -> None:
    mock = MockAnthropic({"needs_retrieval": True, "standalone_query": "q"})
    rewriter, budget = make_rewriter(mock, tmp_path, limit=1)
    await rewriter.rewrite(QUESTION, HISTORY)
    with pytest.raises(LLMBudgetExhaustedError, match="LLM budget exhausted"):
        await rewriter.rewrite(QUESTION, HISTORY)
    assert len(mock.requests) == 1
    assert budget.used() == 1


async def test_budget_running_out_mid_retry_stops_the_retry(tmp_path: Path) -> None:
    mock = MockAnthropic({"needs_retrieval": True, "standalone_query": ""})
    rewriter, budget = make_rewriter(mock, tmp_path, limit=1, max_retries=3)
    with pytest.raises(LLMBudgetExhaustedError):
        await rewriter.rewrite(QUESTION, HISTORY)
    assert len(mock.requests) == 1  # the retry was refused by the budget, not sent
    assert budget.used() == 1


async def test_transient_api_error_is_retried_once_and_every_attempt_is_counted(
    tmp_path: Path,
) -> None:
    mock = MockAnthropic(529, {"needs_retrieval": True, "standalone_query": "q"})
    rewriter, budget = make_rewriter(mock, tmp_path)
    result = await rewriter.rewrite(QUESTION, HISTORY)
    assert result.standalone_query == "q"
    assert len(mock.requests) == 2
    assert budget.used() == 2  # the SDK retry is a real request, so it is counted


async def test_persistent_api_error_raises_llm_error_with_status(tmp_path: Path) -> None:
    mock = MockAnthropic(500)
    rewriter, budget = make_rewriter(mock, tmp_path)
    with pytest.raises(LLMError, match="HTTP 500") as excinfo:
        await rewriter.rewrite(QUESTION, HISTORY)
    assert not isinstance(excinfo.value, LLMTimeoutError)
    assert excinfo.value.details["status_code"] == 500
    assert len(mock.requests) == 2  # first attempt + one transport retry
    assert budget.used() == 2


async def test_non_retryable_api_error_is_sent_once(tmp_path: Path) -> None:
    mock = MockAnthropic(400)
    rewriter, budget = make_rewriter(mock, tmp_path)
    with pytest.raises(LLMError, match="HTTP 400"):
        await rewriter.rewrite(QUESTION, HISTORY)
    assert len(mock.requests) == 1
    assert budget.used() == 1


async def test_budget_spent_during_transport_retry_is_reported_as_exhausted(
    tmp_path: Path,
) -> None:
    mock = MockAnthropic(529)
    rewriter, budget = make_rewriter(mock, tmp_path, limit=1)
    with pytest.raises(LLMBudgetExhaustedError):
        await rewriter.rewrite(QUESTION, HISTORY)
    assert len(mock.requests) == 1
    assert budget.used() == 1


async def test_timeout_maps_to_llm_timeout_error(tmp_path: Path) -> None:
    mock = MockAnthropic(httpx.ReadTimeout("timed out"))
    rewriter, budget = make_rewriter(mock, tmp_path)
    with pytest.raises(LLMTimeoutError):
        await rewriter.rewrite(QUESTION, HISTORY)
    assert budget.used() == 2  # timeouts are retried once by the SDK; both attempts counted
    await rewriter.aclose()
