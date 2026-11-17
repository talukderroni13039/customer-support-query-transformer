from typing import Any

import pytest

from src.core.exceptions import RerankerUnavailableError
from src.core.lifespan import AppResources
from src.core.schemas.chunk import ChunkHit
from src.domains.support_search.schemas import ArticleCreate, RetrievalMode, ScoreType
from src.domains.support_search.services.article_service import ArticleService
from src.domains.support_search.services.retrieval_service import RetrievalService
from tests.fakes import FakeReranker, Spy

ARTICLES = {
    "KB-101": ("Trailblazer Pro Jacket", "The TB-PRO-22 is a fully waterproof shell jacket."),
    "KB-102": ("Trailblazer Lite Jacket", "The TB-PRO-23 is a water resistant shell jacket."),
    "KB-012": ("Returns for Outerwear", "Jackets can be returned within 60 days of delivery."),
    "KB-031": ("Shipping to Canada", "Standard shipping to Canada takes 6 to 10 business days."),
}


@pytest.fixture
async def seeded(resources: AppResources) -> AppResources:
    service = ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
    )
    for article_id, (title, content) in ARTICLES.items():
        await service.upsert(article_id, ArticleCreate(title=title, content=content))
    return resources


@pytest.fixture
def spies(seeded: AppResources) -> tuple[Spy, Spy, Spy]:
    return Spy(seeded.embedder), Spy(seeded.vector_store), Spy(seeded.keyword_index)


def make_service(spies: tuple[Spy, Spy, Spy], **kwargs: Any) -> RetrievalService:
    embedder, vector_store, keyword_index = spies
    return RetrievalService(
        embedder=embedder,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        keyword_index=keyword_index,  # type: ignore[arg-type]
        **kwargs,
    )


async def test_vector_never_calls_bm25(spies: tuple[Spy, Spy, Spy]) -> None:
    embedder, vector_store, keyword_index = spies
    outcome = await make_service(spies).retrieve("shipping to canada", RetrievalMode.VECTOR)
    assert keyword_index.calls == []
    assert embedder.called("embed_query") == 1
    assert vector_store.called("query") == 1
    assert outcome.score_type is ScoreType.COSINE
    assert outcome.chunks[0].hit.article_id == "KB-031"
    assert [c.vector_rank for c in outcome.chunks] == [1, 2, 3, 4]
    assert all(c.keyword_rank is None and c.rrf_score is None for c in outcome.chunks)


async def test_keyword_never_calls_chroma_or_embedder(spies: tuple[Spy, Spy, Spy]) -> None:
    embedder, vector_store, keyword_index = spies
    outcome = await make_service(spies).retrieve(
        "Is the TB-PRO-23 a jacket?", RetrievalMode.KEYWORD
    )
    assert embedder.calls == []
    assert vector_store.calls == []
    assert keyword_index.called("search") == 1
    assert outcome.score_type is ScoreType.BM25
    assert outcome.chunks[0].hit.article_id == "KB-102"
    assert outcome.chunks[0].keyword_rank == 1
    assert all(c.vector_rank is None and c.rrf_score is None for c in outcome.chunks)


async def test_hybrid_calls_both_and_fuses_with_rrf(spies: tuple[Spy, Spy, Spy]) -> None:
    _, vector_store, keyword_index = spies
    outcome = await make_service(spies).retrieve(
        "TB-PRO-22 waterproof jacket", RetrievalMode.HYBRID
    )
    assert vector_store.called("query") == 1
    assert keyword_index.called("search") == 1
    assert outcome.score_type is ScoreType.RRF
    assert outcome.rerank_fallback is False

    top = outcome.chunks[0]
    assert top.hit.article_id == "KB-101"
    assert top.vector_rank is not None
    assert top.keyword_rank == 1
    expected = 1 / (60 + top.vector_rank) + 1 / (60 + top.keyword_rank)
    assert top.rrf_score == pytest.approx(expected)
    assert top.hit.score == top.rrf_score
    scores = [c.hit.score for c in outcome.chunks]
    assert scores == sorted(scores, reverse=True)
    # A chunk only the vector list found still appears, with no keyword rank.
    vector_only = [c for c in outcome.chunks if c.keyword_rank is None]
    assert vector_only
    assert all(c.vector_rank is not None for c in vector_only)


async def test_candidates_per_retriever_and_rrf_k_are_applied(spies: tuple[Spy, Spy, Spy]) -> None:
    _, vector_store, keyword_index = spies
    service = make_service(spies, candidates_per_retriever=2, rrf_k=10)
    outcome = await service.retrieve("shell jacket", RetrievalMode.HYBRID)
    assert vector_store.calls[-1][1][1] == 2
    assert keyword_index.calls[-1][1][1] == 2
    assert len(outcome.chunks) <= 4
    assert outcome.chunks[0].rrf_score is not None
    assert outcome.chunks[0].rrf_score <= 2 / 11


async def test_query_with_no_keyword_match_still_returns_vector_hits(
    spies: tuple[Spy, Spy, Spy],
) -> None:
    service = make_service(spies)
    hybrid = await service.retrieve("zzzz qqqq", RetrievalMode.HYBRID)
    assert len(hybrid.chunks) == 4
    assert all(c.keyword_rank is None for c in hybrid.chunks)
    keyword = await service.retrieve("zzzz qqqq", RetrievalMode.KEYWORD)
    assert keyword.chunks == []


# --- reranking -------------------------------------------------------------------------------


async def test_rerank_false_never_calls_the_reranker(spies: tuple[Spy, Spy, Spy]) -> None:
    reranker = FakeReranker()
    service = make_service(spies, reranker=reranker)
    outcome = await service.retrieve("shell jacket", RetrievalMode.HYBRID, rerank=False)
    assert reranker.calls == []
    assert outcome.score_type is ScoreType.RRF
    assert all(c.rerank_score is None for c in outcome.chunks)


async def test_reranker_receives_at_most_rerank_candidates(spies: tuple[Spy, Spy, Spy]) -> None:
    reranker = FakeReranker()
    service = make_service(spies, reranker=reranker, rerank_candidates=2)
    before = await service.retrieve("shell jacket", RetrievalMode.HYBRID, rerank=False)
    outcome = await service.retrieve("shell jacket", RetrievalMode.HYBRID, rerank=True)
    query, candidates = reranker.calls[0]
    assert query == "shell jacket"
    assert len(candidates) == 2
    assert [c.chunk_id for c in candidates] == [c.hit.chunk_id for c in before.chunks[:2]]
    assert len(outcome.chunks) == 2  # only the reranked candidates are returned


async def test_rerank_returns_same_items_reordered_with_scores_in_0_1(
    spies: tuple[Spy, Spy, Spy],
) -> None:
    service = make_service(spies, reranker=FakeReranker())
    before = await service.retrieve("returned within 60 days", RetrievalMode.VECTOR, rerank=False)
    after = await service.retrieve("returned within 60 days", RetrievalMode.VECTOR, rerank=True)
    assert {c.hit.chunk_id for c in after.chunks} == {c.hit.chunk_id for c in before.chunks}
    assert after.chunks[0].hit.article_id == "KB-012"
    assert all(0.0 <= c.hit.score <= 1.0 for c in after.chunks)
    scores = [c.hit.score for c in after.chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(c.rerank_score == c.hit.score for c in after.chunks)
    assert all(c.vector_rank is not None for c in after.chunks)  # earlier stage debug is kept


@pytest.mark.parametrize(
    ("mode", "rerank", "expected"),
    [
        (RetrievalMode.VECTOR, False, ScoreType.COSINE),
        (RetrievalMode.KEYWORD, False, ScoreType.BM25),
        (RetrievalMode.HYBRID, False, ScoreType.RRF),
        (RetrievalMode.VECTOR, True, ScoreType.RERANK),
        (RetrievalMode.KEYWORD, True, ScoreType.RERANK),
        (RetrievalMode.HYBRID, True, ScoreType.RERANK),
    ],
)
async def test_score_type_for_every_combination(
    spies: tuple[Spy, Spy, Spy], mode: RetrievalMode, rerank: bool, expected: ScoreType
) -> None:
    service = make_service(spies, reranker=FakeReranker())
    outcome = await service.retrieve("waterproof shell jacket", mode, rerank=rerank)
    assert outcome.score_type is expected
    assert outcome.chunks


async def test_hybrid_rerank_keeps_rrf_debug(spies: tuple[Spy, Spy, Spy]) -> None:
    service = make_service(spies, reranker=FakeReranker())
    outcome = await service.retrieve("TB-PRO-22 waterproof", RetrievalMode.HYBRID, rerank=True)
    top = outcome.chunks[0]
    assert top.rrf_score is not None
    assert top.rerank_score is not None
    assert top.keyword_rank == 1


async def test_reranker_failure_keeps_fused_order_and_flags_fallback(
    spies: tuple[Spy, Spy, Spy],
) -> None:
    reranker = FakeReranker()
    reranker.fail = True
    service = make_service(spies, reranker=reranker)
    plain = await service.retrieve("shell jacket", RetrievalMode.HYBRID, rerank=False)
    outcome = await service.retrieve("shell jacket", RetrievalMode.HYBRID, rerank=True)
    assert outcome.rerank_fallback is True
    assert outcome.score_type is ScoreType.RRF
    assert [c.hit.chunk_id for c in outcome.chunks] == [c.hit.chunk_id for c in plain.chunks]
    assert all(c.rerank_score is None for c in outcome.chunks)


async def test_reranker_returning_wrong_items_is_treated_as_failure(
    spies: tuple[Spy, Spy, Spy],
) -> None:
    class DroppingReranker(FakeReranker):
        def rerank(self, query: str, candidates: list[ChunkHit]) -> list[ChunkHit]:
            return super().rerank(query, candidates)[:-1]

    service = make_service(spies, reranker=DroppingReranker())
    outcome = await service.retrieve("shell jacket", RetrievalMode.VECTOR, rerank=True)
    assert outcome.rerank_fallback is True
    assert outcome.score_type is ScoreType.COSINE


async def test_rerank_requested_without_reranker_raises(spies: tuple[Spy, Spy, Spy]) -> None:
    service = make_service(spies)
    assert service.rerank_available is False
    with pytest.raises(RerankerUnavailableError):
        await service.retrieve("shell jacket", RetrievalMode.HYBRID, rerank=True)


async def test_rerank_with_no_candidates_skips_the_reranker(spies: tuple[Spy, Spy, Spy]) -> None:
    reranker = FakeReranker()
    service = make_service(spies, reranker=reranker)
    outcome = await service.retrieve("zzzz qqqq", RetrievalMode.KEYWORD, rerank=True)
    assert outcome.chunks == []
    assert reranker.calls == []
    assert outcome.rerank_fallback is False
