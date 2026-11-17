"""Real-model smoke tests. Excluded by default; run once with: uv run pytest -m model --no-cov

Downloads `BAAI/bge-small-en-v1.5` (~130 MB) and `BAAI/bge-reranker-base` (~1.1 GB) on first run.
"""

import math

import pytest

from src.core.schemas.chunk import ChunkHit
from src.embeddings.bge_embedder import BgeEmbedder
from src.reranking.bge_reranker import BgeReranker

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> BgeEmbedder:
    return BgeEmbedder()


@pytest.fixture(scope="module")
def reranker() -> BgeReranker:
    return BgeReranker()


def test_embedder_has_384_normalized_dimensions(embedder: BgeEmbedder) -> None:
    assert embedder.dimensions == 384
    vectors = embedder.embed_documents(["Returns for Outerwear", "Shipping to Canada"])
    query = embedder.embed_query("How do I send a jacket back?")
    for vector in [*vectors, query]:
        assert len(vector) == 384
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-3)


def test_embedder_ranks_the_related_text_closer(embedder: BgeEmbedder) -> None:
    returns, shipping = embedder.embed_documents(
        ["Jackets can be returned within 60 days of delivery.", "Shipping to Canada takes 6 days."]
    )
    query = embedder.embed_query("Can I send my coat back?")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, returns) > cosine(query, shipping)


def test_reranker_scores_relevant_pair_higher_than_irrelevant(reranker: BgeReranker) -> None:
    candidates = [
        ChunkHit(
            chunk_id="KB-031::0000",
            article_id="KB-031",
            title="Shipping to Canada",
            heading="Delivery times",
            text="Standard shipping to Canada takes 6 to 10 business days.",
            score=0.9,
        ),
        ChunkHit(
            chunk_id="KB-012::0000",
            article_id="KB-012",
            title="Returns for Outerwear",
            heading="Return window",
            text="Jackets can be returned within 60 days of delivery.",
            score=0.1,
        ),
    ]
    result = reranker.rerank("How long do I have to return a jacket?", candidates)
    assert {h.chunk_id for h in result} == {h.chunk_id for h in candidates}  # same items
    assert result[0].chunk_id == "KB-012::0000"  # re-ordered
    assert all(0.0 <= h.score <= 1.0 for h in result)
    assert result[0].score > result[1].score
