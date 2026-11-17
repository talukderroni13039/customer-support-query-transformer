"""Cross-encoder reranking with sentence-transformers (`BAAI/bge-reranker-base` by default)."""

from typing import Any

from src.core.schemas.chunk import ChunkHit

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class BgeReranker:
    """Synchronous and CPU-bound: callers must run it in a worker thread.

    Scores `(query, "{title} — {heading}: {chunk}")` pairs, maps the logit through a sigmoid
    to 0-1, and returns the same candidates sorted best first. It never picks candidates.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        *,
        model: Any | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device=device, max_length=max_length)
        self._model = model
        self._batch_size = batch_size
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[ChunkHit]) -> list[ChunkHit]:
        if not candidates:
            return []
        import torch

        scores = self._model.predict(
            [(query, hit.index_text) for hit in candidates],
            batch_size=self._batch_size,
            activation_fn=torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
        rescored = [
            hit.with_score(float(score)) for hit, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)
