from typing import Any

import numpy as np

from src.core.schemas.chunk import ChunkHit
from src.reranking.bge_reranker import BgeReranker


class StubCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any:
        assert kwargs["activation_fn"] is not None  # sigmoid → scores in 0-1
        self.pairs = list(pairs)
        return np.array(self.scores[: len(pairs)], dtype=np.float32)


def hit(chunk_id: str, text: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        article_id=chunk_id.split("::")[0],
        title="Returns",
        heading="Window",
        text=f"## Window\n{text}",
        score=0.5,
    )


def test_rerank_scores_title_heading_chunk_pairs_and_sorts() -> None:
    model = StubCrossEncoder([0.2, 0.9, 0.5])
    reranker = BgeReranker("stub-reranker", model=model)
    candidates = [hit("a::0", "alpha"), hit("b::0", "beta"), hit("c::0", "gamma")]

    result = reranker.rerank("return window", candidates)

    assert reranker.model_name == "stub-reranker"
    assert model.pairs[0] == ("return window", "Returns — Window: alpha")
    assert [h.chunk_id for h in result] == ["b::0", "c::0", "a::0"]
    assert [round(h.score, 2) for h in result] == [0.9, 0.5, 0.2]
    assert all(isinstance(h.score, float) for h in result)
    assert {h.chunk_id for h in result} == {h.chunk_id for h in candidates}
    assert candidates[0].score == 0.5  # inputs are not mutated


def test_rerank_with_no_candidates_does_not_call_the_model() -> None:
    model = StubCrossEncoder([])
    assert BgeReranker("stub-reranker", model=model).rerank("q", []) == []
    assert model.pairs == []
