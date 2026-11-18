from typing import Any

import numpy as np

from src.embeddings.bge_embedder import BGE_QUERY_INSTRUCTION, BgeEmbedder


class StubModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        assert kwargs["normalize_embeddings"] is True
        self.calls.append(list(texts))
        return np.full((len(texts), 4), 0.5, dtype=np.float32)


def test_embedder_reports_dimensions_and_prefixes_queries_only() -> None:
    model = StubModel()
    embedder = BgeEmbedder("stub-model", model=model)
    assert embedder.dimensions == 4
    assert embedder.model_name == "stub-model"

    docs = embedder.embed_documents(["a", "b"])
    assert docs == [[0.5] * 4, [0.5] * 4]
    assert all(isinstance(x, float) for x in docs[0])
    assert model.calls[-1] == ["a", "b"]

    assert embedder.embed_query("q") == [0.5] * 4
    assert model.calls[-1] == [BGE_QUERY_INSTRUCTION + "q"]

    calls_before = len(model.calls)
    assert embedder.embed_documents([]) == []
    assert len(model.calls) == calls_before
