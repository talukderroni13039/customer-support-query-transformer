import json
import threading
from pathlib import Path

import pytest

from src.core.exceptions import LLMBudgetExhaustedError
from src.llm.call_budget import LlmCallBudget


def test_counts_persist_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "llm_call_budget.json"
    budget = LlmCallBudget(path, limit=3)
    assert budget.used() == 0
    assert budget.remaining() == 3
    assert budget.consume() == 1
    assert budget.consume() == 2

    reopened = LlmCallBudget(path, limit=3)
    assert reopened.used() == 2
    assert reopened.remaining() == 1
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["calls"] == 2
    assert saved["budget"] == 3


def test_refuses_at_the_limit_and_never_goes_over(tmp_path: Path) -> None:
    budget = LlmCallBudget(tmp_path / "b.json", limit=2)
    budget.consume()
    budget.ensure_available()
    budget.consume()
    with pytest.raises(LLMBudgetExhaustedError, match="LLM budget exhausted"):
        budget.consume()
    with pytest.raises(LLMBudgetExhaustedError, match="LLM budget exhausted"):
        budget.ensure_available()
    assert budget.used() == 2
    assert budget.remaining() == 0


def test_unreadable_or_invalid_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    for content in ("not json", "{}", '{"calls": "many"}', '{"calls": -1}', "[1]"):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(LLMBudgetExhaustedError):
            LlmCallBudget(path, limit=50).consume()


def test_concurrent_consumers_never_exceed_the_limit(tmp_path: Path) -> None:
    budget = LlmCallBudget(tmp_path / "b.json", limit=20)
    granted: list[int] = []
    refused: list[int] = []

    def worker() -> None:
        for _ in range(10):
            try:
                granted.append(budget.consume())
            except LLMBudgetExhaustedError:
                refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(granted) == list(range(1, 21))
    assert len(refused) == 30
    assert budget.used() == 20
