"""Local embeddings with sentence-transformers (`BAAI/bge-small-en-v1.5` by default)."""

from typing import Any

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE v1.5 retrieval convention: instruct the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BgeEmbedder:
    """Synchronous and CPU-bound: callers must run it in a worker thread."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        *,
        model: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
        self._model = model
        self._batch_size = batch_size
        self.model_name = model_name
        self.dimensions = len(self._encode(["dimension probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([BGE_QUERY_INSTRUCTION + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]
