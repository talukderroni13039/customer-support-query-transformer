"""Live rewrite tests: REAL Anthropic calls. Excluded by default.

Run once (PowerShell):  uv run pytest -m live --no-cov

Exactly 5 cases, `max_retries=1`. Every attempt is counted in `data/cache/llm_call_budget.json`
against the 50-call cap for the whole build. Do not use these to debug; fix with fakes first.
"""

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.llm.anthropic_rewriter import AnthropicRewriter
from src.llm.call_budget import DEFAULT_LLM_CALL_BUDGET, LlmCallBudget

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rewriter() -> Iterator[AnthropicRewriter]:
    load_dotenv(ROOT / ".env.dev", override=False)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set (.env.dev)")
    budget = LlmCallBudget(
        ROOT / "data" / "cache" / "llm_call_budget.json", DEFAULT_LLM_CALL_BUDGET
    )
    yield AnthropicRewriter(api_key=api_key, budget=budget, max_retries=1)
    os.environ.pop("ANTHROPIC_API_KEY", None)


async def test_pronoun(rewriter: AnthropicRewriter) -> None:
    result = await rewriter.rewrite(
        "What does its warranty cover?",
        ["I'm thinking about the Basecamp 2 tent.", "The Basecamp 2 (BC-200) is a great choice."],
    )
    assert result.needs_retrieval is True
    query = result.standalone_query.lower()
    assert "warranty" in query
    assert "basecamp 2" in query or "bc-200" in query


async def test_ellipsis(rewriter: AnthropicRewriter) -> None:
    result = await rewriter.rewrite(
        "What about the EU?",
        [
            "How much is shipping to Australia?",
            "Shipping to Australia costs a flat $30.",
        ],
    )
    assert result.needs_retrieval is True
    query = result.standalone_query.lower()
    assert "shipping" in query, query
    assert re.search(r"\beu\b|europe", query), query


async def test_correction(rewriter: AnthropicRewriter) -> None:
    result = await rewriter.rewrite(
        "No sorry, I meant shoes, not a jacket.",
        ["How do I choose the right size jacket?", "Measure your chest and check the size chart."],
    )
    assert result.needs_retrieval is True
    query = result.standalone_query.lower()
    assert "siz" in query
    assert "shoe" in query or "footwear" in query
    assert "jacket" not in query


async def test_small_talk(rewriter: AnthropicRewriter) -> None:
    result = await rewriter.rewrite(
        "Perfect, thanks a lot!",
        ["Can I pay with Apple Pay?", "Yes, Apple Pay is accepted at checkout."],
    )
    assert result.needs_retrieval is False


async def test_sku_preserved_exactly(rewriter: AnthropicRewriter) -> None:
    result = await rewriter.rewrite(
        "Can I put it in the washing machine?",
        [
            "Is the Trailblazer Pro waterproof?",
            "Yes, the Trailblazer Pro (TB-PRO-22) is waterproof.",
        ],
    )
    assert result.needs_retrieval is True
    assert "TB-PRO-22" in result.standalone_query
    assert "TB-PRO-23" not in result.standalone_query
