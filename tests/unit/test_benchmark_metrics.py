import pytest

from scripts.run_benchmark import (
    CONFIGS,
    END_MARKER,
    START_MARKER,
    CaseResult,
    first_correct_rank,
    hit_at_k,
    mean_reciprocal_rank,
    percentile,
    render_tables,
    small_talk_correct,
)


def result(
    ranked: list[str],
    expected: list[str],
    *,
    pattern: str = "pronoun",
    config: int = 1,
    needs_retrieval: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id="C-1",
        pattern=pattern,
        config=config,
        search_query="q",
        ranked_ids=ranked,
        expected_ids=expected,
        needs_retrieval=needs_retrieval,
        rewrite_source="none",
        rewrite_fallback=False,
        rerank_fallback=False,
        latency_ms=10.0,
    )


def test_first_correct_rank() -> None:
    assert first_correct_rank(["a", "b", "c"], ["c", "b"]) == 2
    assert first_correct_rank(["a"], ["z"]) is None
    assert first_correct_rank([], ["z"]) is None


def test_hit_at_k_and_mrr() -> None:
    rows = [
        result(["a", "b", "c"], ["a"]),  # rank 1
        result(["a", "b", "c"], ["c"]),  # rank 3
        result(["a", "b", "c", "d"], ["d"]),  # rank 4
        result(["a"], ["z"]),  # missing
    ]
    assert hit_at_k(rows, 1) == 0.25
    assert hit_at_k(rows, 3) == 0.5
    assert mean_reciprocal_rank(rows) == pytest.approx((1 + 1 / 3 + 1 / 4 + 0) / 4)
    assert hit_at_k([], 1) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_percentile_and_small_talk() -> None:
    assert percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 95) == 100
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([], 95) == 0.0
    talk = [
        result([], [], pattern="small_talk", needs_retrieval=False),
        result(["a"], [], pattern="small_talk", needs_retrieval=True),
    ]
    assert small_talk_correct(talk) == 1


def test_render_tables_has_six_configs_and_per_pattern_rows() -> None:
    rows = []
    for config in CONFIGS:
        rows.append(result(["a"], ["a"], pattern="pronoun", config=config.number))
        rows.append(result(["b"], ["a"], pattern="exact_code", config=config.number))
        rows.append(
            result([], [], pattern="small_talk", config=config.number, needs_retrieval=False)
        )
    meta = {
        "run_at": "now",
        "cases": 3,
        "retrieval_cases": 2,
        "small_talk_cases": 1,
        "articles": 1,
        "chunks": 1,
        "embedding_model": "e",
        "reranker_model": "r",
        "rewrite_provider": "anthropic",
        "candidates": 20,
        "rrf_k": 60,
        "rerank_candidates": 10,
        "max_history": 12,
        "rerank_min_score": 0.01,
        "llm_calls_this_run": 0,
        "llm_calls_total": 11,
        "llm_call_budget": 50,
    }
    text = render_tables(rows, meta)
    assert text.startswith(START_MARKER)
    assert text.endswith(END_MARKER)
    for config in CONFIGS:
        assert f"| {config.number} | " in text
    assert "| `pronoun` | 1 | 1.00 |" in text
    assert "| `exact_code` | 1 | 0.00 |" in text
    assert "`small_talk`" not in text  # scored separately, in the configuration table
    assert "1/1" in text
