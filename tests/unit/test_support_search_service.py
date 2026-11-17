import pytest

from src.core.exceptions import (
    KnowledgeBaseEmptyError,
    LLMError,
    LLMTimeoutError,
    RerankerUnavailableError,
)
from src.core.lifespan import AppResources
from src.core.schemas.chunk import ChunkHit
from src.domains.support_search.schemas import (
    ArticleCreate,
    QueryMode,
    RetrievalMode,
    ScoreType,
    SearchRequest,
)
from src.domains.support_search.services.article_service import ArticleService
from src.domains.support_search.services.query_service import QueryService
from src.domains.support_search.services.retrieval_service import RankedChunk, RetrievalService
from src.domains.support_search.services.support_search_service import (
    SNIPPET_MAX_CHARS,
    SupportSearchService,
    group_by_article,
    make_snippet,
)
from tests.fakes import FakeRewriter

QUESTION = "How long do I have to return it?"
HISTORY = [
    "Do you sell the Trailblazer jacket in waterproof?",
    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.",
]
ARTICLES = {
    "KB-101": (
        "Trailblazer Pro Jacket",
        "## Overview\nThe TB-PRO-22 is a waterproof shell jacket.",
    ),
    "KB-012": (
        "Returns for Outerwear",
        "## Return window\nJackets such as the TB-PRO-22 can be returned within 60 days.\n\n"
        "## Condition\nJackets must be unworn with tags attached.",
    ),
    "KB-011": ("Returns for Footwear", "## Return window\nBoots can be returned within 30 days."),
    "KB-031": ("Shipping to Canada", "Standard shipping to Canada takes 6 to 10 business days."),
}


@pytest.fixture
async def seeded(resources: AppResources) -> AppResources:
    articles = ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
    )
    for article_id, (title, content) in ARTICLES.items():
        await articles.upsert(article_id, ArticleCreate(title=title, content=content))
    return resources


def make_service(
    resources: AppResources, rewriter: FakeRewriter | None = None, strict: bool = False
) -> SupportSearchService:
    return SupportSearchService(
        query_service=QueryService(rewriter, strict=strict),
        retrieval_service=RetrievalService(
            embedder=resources.embedder,
            vector_store=resources.vector_store,
            keyword_index=resources.keyword_index,
            reranker=resources.reranker,
        ),
        keyword_index=resources.keyword_index,
    )


async def test_rewrite_mode_searches_with_the_standalone_query(seeded: AppResources) -> None:
    rewriter = FakeRewriter("Return window for the Trailblazer Pro TB-PRO-22 jacket")
    result = await make_service(seeded, rewriter).search(
        SearchRequest(question=QUESTION, history=HISTORY)
    )
    assert result.question == QUESTION
    assert result.search_query == "Return window for the Trailblazer Pro TB-PRO-22 jacket"
    assert result.query_mode is QueryMode.REWRITE
    assert result.rewrite_source == "llm"
    assert result.rewrite_fallback is False
    assert result.needs_retrieval is True
    assert result.score_type is ScoreType.RERANK
    assert result.results[0].article_id == "KB-012"
    assert seeded.embedder.query_calls[-1] == result.search_query  # type: ignore[attr-defined]


async def test_question_only_and_full_history_never_call_the_rewriter(
    seeded: AppResources,
) -> None:
    rewriter = FakeRewriter()
    service = make_service(seeded, rewriter)
    naive = await service.search(
        SearchRequest(question=QUESTION, history=HISTORY, query_mode="question_only")
    )
    assert naive.search_query == QUESTION
    assert naive.rewrite_source == "none"

    full = await service.search(
        SearchRequest(question=QUESTION, history=HISTORY, query_mode="full_history")
    )
    assert full.search_query == "\n".join([*HISTORY, QUESTION])
    assert rewriter.calls == []


async def test_empty_history_skips_the_rewriter(seeded: AppResources) -> None:
    rewriter = FakeRewriter()
    result = await make_service(seeded, rewriter).search(
        SearchRequest(question="shipping to Canada")
    )
    assert result.rewrite_source == "skipped"
    assert result.search_query == "shipping to Canada"
    assert result.results[0].article_id == "KB-031"
    assert rewriter.calls == []


async def test_small_talk_returns_empty_results_without_retrieval(seeded: AppResources) -> None:
    rewriter = FakeRewriter(needs_retrieval=False)
    queries_before = len(seeded.embedder.query_calls)  # type: ignore[attr-defined]
    result = await make_service(seeded, rewriter).search(
        SearchRequest(question="Thanks, that's all!", history=HISTORY)
    )
    assert result.needs_retrieval is False
    assert result.results == []
    assert result.search_query == ""
    assert result.rewrite_fallback is False
    assert len(seeded.embedder.query_calls) == queries_before  # type: ignore[attr-defined]
    assert seeded.reranker.calls == []  # type: ignore[union-attr]


async def test_rewrite_failure_falls_back_to_question_and_still_searches(
    seeded: AppResources,
) -> None:
    rewriter = FakeRewriter(error=LLMTimeoutError())
    result = await make_service(seeded, rewriter).search(
        SearchRequest(question=QUESTION, history=HISTORY)
    )
    assert result.rewrite_fallback is True
    assert result.rewrite_source == "fallback"
    assert result.search_query == QUESTION
    assert result.results  # the search itself never fails


async def test_strict_mode_raises_llm_errors(seeded: AppResources) -> None:
    service = make_service(seeded, FakeRewriter(error=LLMTimeoutError()), strict=True)
    with pytest.raises(LLMTimeoutError):
        await service.search(SearchRequest(question=QUESTION, history=HISTORY))
    service = make_service(seeded, FakeRewriter(error=ValueError("x")), strict=True)
    with pytest.raises(LLMError):
        await service.search(SearchRequest(question=QUESTION, history=HISTORY))


async def test_results_are_grouped_by_article_with_best_chunk(seeded: AppResources) -> None:
    result = await make_service(seeded).search(
        SearchRequest(
            question="jackets returned unworn tags 60 days",
            query_mode="question_only",
            retrieval="keyword",
            rerank=False,
            top_k=10,
            include_debug=True,
        )
    )
    ids = [r.article_id for r in result.results]
    assert len(ids) == len(set(ids))  # KB-012 has two matching chunks but appears once
    assert ids[0] == "KB-012"
    assert result.results[0].debug is not None
    assert result.results[0].debug.keyword_rank == 1
    assert result.score_type is ScoreType.BM25


async def test_top_k_default_and_override(seeded: AppResources) -> None:
    service = make_service(seeded)
    base = {
        "question": "return jacket boots shipping",
        "query_mode": "question_only",
        "rerank": False,
    }
    assert len((await service.search(SearchRequest(**base))).results) == 3  # type: ignore[arg-type]
    assert len((await service.search(SearchRequest(**base, top_k=1))).results) == 1  # type: ignore[arg-type]
    assert len((await service.search(SearchRequest(**base, top_k=10))).results) == 4  # type: ignore[arg-type]


async def test_reranker_fallback_is_reported(seeded: AppResources) -> None:
    seeded.reranker.fail = True  # type: ignore[union-attr]
    result = await make_service(seeded).search(
        SearchRequest(question="waterproof jacket", query_mode="question_only")
    )
    assert result.rerank is True
    assert result.rerank_fallback is True
    assert result.score_type is ScoreType.RRF
    assert result.results


async def test_failing_preconditions_never_cost_an_llm_call(resources: AppResources) -> None:
    rewriter = FakeRewriter()
    with pytest.raises(KnowledgeBaseEmptyError):
        await make_service(resources, rewriter).search(
            SearchRequest(question=QUESTION, history=HISTORY)
        )
    articles = ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
    )
    await articles.upsert(
        "KB-1", ArticleCreate(title="Title", content="Some article content here.")
    )
    resources.reranker = None
    with pytest.raises(RerankerUnavailableError):
        await make_service(resources, rewriter).search(
            SearchRequest(question=QUESTION, history=HISTORY)
        )
    assert rewriter.calls == []


@pytest.mark.parametrize(
    ("retrieval", "rerank", "expected"),
    [
        (RetrievalMode.VECTOR, False, ScoreType.COSINE),
        (RetrievalMode.KEYWORD, False, ScoreType.BM25),
        (RetrievalMode.HYBRID, False, ScoreType.RRF),
        (RetrievalMode.HYBRID, True, ScoreType.RERANK),
    ],
)
async def test_small_talk_still_reports_the_score_type_of_the_switches(
    seeded: AppResources, retrieval: RetrievalMode, rerank: bool, expected: ScoreType
) -> None:
    result = await make_service(seeded, FakeRewriter(needs_retrieval=False)).search(
        SearchRequest(question="Thanks!", history=HISTORY, retrieval=retrieval, rerank=rerank)
    )
    assert result.score_type is expected


def _chunk(chunk_id: str, article_id: str, text: str, heading: str = "") -> RankedChunk:
    return RankedChunk(
        hit=ChunkHit(
            chunk_id=chunk_id,
            article_id=article_id,
            title="T",
            heading=heading,
            text=text,
            score=0.5,
        ),
        vector_rank=1,
    )


def test_group_by_article_keeps_first_chunk_per_article_and_optional_debug() -> None:
    chunks = [_chunk("a::0", "a", "first"), _chunk("a::1", "a", "second"), _chunk("b::0", "b", "x")]
    grouped = group_by_article(chunks, top_k=5, include_debug=False)
    assert [(g.article_id, g.snippet) for g in grouped] == [("a", "first"), ("b", "x")]
    assert grouped[0].debug is None
    with_debug = group_by_article(chunks, top_k=1, include_debug=True)
    assert len(with_debug) == 1
    assert with_debug[0].debug is not None
    assert with_debug[0].debug.vector_rank == 1
    assert group_by_article([], top_k=3, include_debug=True) == []


def test_snippet_drops_heading_line_and_truncates() -> None:
    assert make_snippet(_chunk("a::0", "a", "## Window\nWithin  60\ndays.\n\n", "Window")) == (
        "Within 60 days."
    )
    long_snippet = make_snippet(_chunk("a::0", "a", "word " * 200))
    assert len(long_snippet) <= SNIPPET_MAX_CHARS + 1
    assert long_snippet.endswith("…")
