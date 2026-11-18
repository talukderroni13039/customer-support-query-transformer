import json
from pathlib import Path

import pytest

from src.llm.cached_rewriter import CachedRewriter
from tests.fakes import FakeRewriter

QUESTION = "How long do I have to return it?"
HISTORY = ["Do you sell the Trailblazer jacket?", "Yes, the Trailblazer Pro (TB-PRO-22)."]


def make(inner: FakeRewriter, path: Path, namespace: str = "model:v1") -> CachedRewriter:
    return CachedRewriter(inner, path, namespace)


async def test_second_call_is_served_from_cache(tmp_path: Path) -> None:
    inner = FakeRewriter("Return window for the TB-PRO-22 jacket")
    cached = make(inner, tmp_path / "rewrites.json")

    first = await cached.rewrite(QUESTION, HISTORY)
    second = await cached.rewrite(QUESTION, HISTORY)

    assert len(inner.calls) == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.standalone_query == first.standalone_query
    assert second.needs_retrieval is True


async def test_cache_survives_restart_and_stores_no_user_text(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "rewrites.json"
    await make(FakeRewriter("Return window TB-PRO-22"), path).rewrite(QUESTION, HISTORY)

    raw = path.read_text(encoding="utf-8")
    assert QUESTION not in raw
    assert HISTORY[0] not in raw
    entry = next(iter(json.loads(raw).values()))
    assert entry["standalone_query"] == "Return window TB-PRO-22"
    assert entry["latency_ms"] >= 0

    fresh_inner = FakeRewriter("should not be used")
    reopened = make(fresh_inner, path)
    result = await reopened.rewrite(QUESTION, HISTORY)
    assert result.from_cache is True
    assert result.standalone_query == "Return window TB-PRO-22"
    assert fresh_inner.calls == []
    assert reopened.cached_latency_ms(QUESTION, HISTORY) is not None
    assert reopened.cached_latency_ms("other", HISTORY) is None


async def test_key_depends_on_question_history_and_namespace(tmp_path: Path) -> None:
    path = tmp_path / "rewrites.json"
    inner = FakeRewriter()
    cached = make(inner, path)
    await cached.rewrite(QUESTION, HISTORY)
    await cached.rewrite(QUESTION + "?", HISTORY)
    await cached.rewrite(QUESTION, [*HISTORY, "one more"])
    await cached.rewrite(QUESTION, list(reversed(HISTORY)))
    assert len(inner.calls) == 4

    bumped_inner = FakeRewriter()
    await make(bumped_inner, path, namespace="model:v2").rewrite(QUESTION, HISTORY)
    assert len(bumped_inner.calls) == 1  # new prompt version → old entry not reused


async def test_small_talk_results_are_cached_too(tmp_path: Path) -> None:
    inner = FakeRewriter(needs_retrieval=False)
    cached = make(inner, tmp_path / "rewrites.json")
    await cached.rewrite("Thanks!", HISTORY)
    result = await cached.rewrite("Thanks!", HISTORY)
    assert result.needs_retrieval is False
    assert result.standalone_query == ""
    assert result.from_cache is True
    assert len(inner.calls) == 1


async def test_errors_are_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "rewrites.json"
    inner = FakeRewriter(error=RuntimeError("claude down"))
    cached = make(inner, path)
    with pytest.raises(RuntimeError):
        await cached.rewrite(QUESTION, HISTORY)
    assert not path.exists()
    inner.error = None
    assert (await cached.rewrite(QUESTION, HISTORY)).from_cache is False
    assert len(inner.calls) == 2


async def test_corrupt_cache_file_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "rewrites.json"
    path.write_text("{ not json", encoding="utf-8")
    inner = FakeRewriter()
    assert (await make(inner, path).rewrite(QUESTION, HISTORY)).from_cache is False
    path.write_text("[1, 2]", encoding="utf-8")
    assert (await make(inner, path).rewrite(QUESTION, HISTORY)).from_cache is False
