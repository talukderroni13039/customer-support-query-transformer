from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core.exceptions import LLMBudgetExhaustedError, LLMTimeoutError
from src.core.lifespan import AppResources
from tests.integration.test_articles_api import err, ok

ARTICLES = "/api/v1/support/articles"
SEARCH = "/api/v1/support/search"

KB = [
    ("Trailblazer Pro Jacket", "The TB-PRO-22 is a fully waterproof shell jacket.", "products"),
    ("Trailblazer Lite Jacket", "The TB-PRO-23 is a water resistant shell jacket.", "products"),
    ("Returns for Outerwear", "Jackets can be returned within 60 days of delivery.", "returns"),
    ("Shipping to Canada", "Standard shipping to Canada takes 6 to 10 business days.", "shipping"),
]


@pytest.fixture
def seeded_client(client: TestClient) -> TestClient:
    for title, content, category in KB:
        ok(
            client.post(ARTICLES, json={"title": title, "content": content, "category": category}),
            201,
        )
    return client


def search(client: TestClient, **body: Any) -> Any:
    return client.post(SEARCH, json=body)


def test_search_returns_envelope_and_grouped_results(seeded_client: TestClient) -> None:
    data = ok(
        search(
            seeded_client,
            question="How long does shipping to Canada take?",
            query_mode="question_only",
            retrieval="hybrid",
            rerank=False,
        )
    )
    assert data["question"] == data["search_query"] == "How long does shipping to Canada take?"
    assert data["retrieval"] == "hybrid"
    assert data["score_type"] == "rrf"
    assert len(data["results"]) == 3  # default top_k
    top = data["results"][0]
    assert top["title"] == "Shipping to Canada"
    assert top["category"] == "shipping"
    assert "6 to 10 business days" in top["snippet"]
    assert top["debug"] is None
    assert len({r["article_id"] for r in data["results"]}) == 3


def test_include_debug_and_top_k(seeded_client: TestClient) -> None:
    data = ok(
        search(
            seeded_client,
            question="TB-PRO-22 waterproof",
            query_mode="question_only",
            rerank=False,
            include_debug=True,
            top_k=1,
        )
    )
    assert len(data["results"]) == 1
    debug = data["results"][0]["debug"]
    assert data["results"][0]["title"] == "Trailblazer Pro Jacket"
    assert debug["keyword_rank"] == 1
    assert debug["vector_rank"] >= 1
    assert debug["rrf_score"] == pytest.approx(data["results"][0]["score"])
    assert debug["rerank_score"] is None


def test_new_article_is_immediately_findable_by_keyword_and_vector(client: TestClient) -> None:
    ok(client.post(ARTICLES, json={"title": KB[3][0], "content": KB[3][1]}), 201)
    created = ok(
        client.post(
            ARTICLES,
            json={
                "title": "Snowshoe Rental",
                "content": "Rent MSR-EVO-9 snowshoes at the Northwind flagship store by the day.",
            },
        ),
        201,
    )
    for retrieval, score_type in (("keyword", "bm25"), ("vector", "cosine")):
        data = ok(
            search(
                client,
                question="Can I rent MSR-EVO-9 snowshoes?",
                query_mode="question_only",
                retrieval=retrieval,
                rerank=False,
            )
        )
        assert data["score_type"] == score_type
        assert data["results"][0]["article_id"] == created["id"], retrieval


def test_deleted_article_disappears_from_search(seeded_client: TestClient) -> None:
    listed = ok(seeded_client.get(ARTICLES, params={"category": "shipping"}))
    ok(seeded_client.delete(f"{ARTICLES}/{listed[0]['id']}"))
    for retrieval in ("keyword", "vector", "hybrid"):
        data = ok(
            search(
                seeded_client,
                question="shipping to Canada",
                query_mode="question_only",
                retrieval=retrieval,
                rerank=False,
            )
        )
        assert all(r["title"] != "Shipping to Canada" for r in data["results"])


def test_empty_knowledge_base_returns_409(client: TestClient) -> None:
    err(search(client, question="anything", rerank=False), 409, "KB_EMPTY")


def test_invalid_requests_return_422(seeded_client: TestClient) -> None:
    cases: list[dict[str, Any]] = [
        {"question": "q", "retrieval": "semantic"},
        {"question": "q", "query_mode": "hyde"},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 11},
        {"question": "q", "history": ["h"] * 101},
        {"question": "q", "history": ["x" * 2001]},
        {"question": "q", "history": [{"role": "user", "content": "hi"}]},
        {"question": "q", "role": "user"},
        {},
    ]
    for body in cases:
        err(seeded_client.post(SEARCH, json=body), 422, "VALIDATION_ERROR")


# --- reranking -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retrieval", "rerank", "score_type"),
    [
        ("vector", False, "cosine"),
        ("keyword", False, "bm25"),
        ("hybrid", False, "rrf"),
        ("vector", True, "rerank"),
        ("keyword", True, "rerank"),
        ("hybrid", True, "rerank"),
    ],
)
def test_every_retrieval_and_rerank_combination_returns_correct_score_type(
    seeded_client: TestClient, retrieval: str, rerank: bool, score_type: str
) -> None:
    data = ok(
        search(
            seeded_client,
            question="Is the TB-PRO-22 jacket waterproof?",
            query_mode="question_only",
            retrieval=retrieval,
            rerank=rerank,
            include_debug=True,
        )
    )
    assert data["retrieval"] == retrieval
    assert data["rerank"] is rerank
    assert data["score_type"] == score_type
    assert data["rerank_fallback"] is False
    top = data["results"][0]
    assert top["title"] == "Trailblazer Pro Jacket"
    assert (top["debug"]["rerank_score"] is not None) is rerank
    assert (top["debug"]["vector_rank"] is not None) is (retrieval != "keyword")
    assert (top["debug"]["keyword_rank"] is not None) is (retrieval != "vector")
    assert (top["debug"]["rrf_score"] is not None) is (retrieval == "hybrid")
    if rerank:
        assert 0.0 <= top["score"] <= 1.0


def test_rerank_true_with_reranker_disabled_returns_409(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.reranker = None
    err(
        search(seeded_client, question="jacket", query_mode="question_only"),
        409,
        "RERANKER_DISABLED",
    )
    ok(search(seeded_client, question="jacket", query_mode="question_only", rerank=False))
    assert ok(seeded_client.get("/health"))["reranker_loaded"] is False


def test_reranker_failure_falls_back_and_is_flagged(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.reranker.fail = True  # type: ignore[union-attr]
    data = ok(search(seeded_client, question="shell jacket", query_mode="question_only"))
    assert data["rerank"] is True
    assert data["rerank_fallback"] is True
    assert data["score_type"] == "rrf"
    assert data["results"]
    assert ok(seeded_client.get("/health"))["reranker_loaded"] is True


# --- query modes (full pipeline) -------------------------------------------------------------

HISTORY = [
    "Do you sell the Trailblazer jacket in waterproof?",
    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.",
]
FOLLOW_UP = "How long do I have to return it?"


def test_default_request_runs_rewrite_hybrid_rerank(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.rewriter.standalone_query = "Can jackets be returned within 60 days of delivery"  # type: ignore[union-attr]
    data = ok(search(seeded_client, question=FOLLOW_UP, history=HISTORY))
    assert data["question"] == FOLLOW_UP
    assert data["search_query"] == "Can jackets be returned within 60 days of delivery"
    assert data["query_mode"] == "rewrite"
    assert data["retrieval"] == "hybrid"
    assert data["rerank"] is True
    assert data["needs_retrieval"] is True
    assert data["rewrite_source"] == "llm"
    assert data["rewrite_fallback"] is False
    assert data["rerank_fallback"] is False
    assert data["score_type"] == "rerank"
    assert data["results"][0]["title"] == "Returns for Outerwear"
    assert resources.rewriter.calls == [(FOLLOW_UP, HISTORY)]  # type: ignore[union-attr]


def test_query_mode_baselines(seeded_client: TestClient, resources: AppResources) -> None:
    naive = ok(
        search(seeded_client, question=FOLLOW_UP, history=HISTORY, query_mode="question_only")
    )
    assert naive["search_query"] == FOLLOW_UP
    assert naive["rewrite_source"] == "none"
    full = ok(search(seeded_client, question=FOLLOW_UP, history=HISTORY, query_mode="full_history"))
    assert full["search_query"] == "\n".join([*HISTORY, FOLLOW_UP])
    assert resources.rewriter.calls == []  # type: ignore[union-attr]


def test_empty_history_skips_claude(seeded_client: TestClient, resources: AppResources) -> None:
    data = ok(search(seeded_client, question="shipping to Canada"))
    assert data["rewrite_source"] == "skipped"
    assert data["results"][0]["title"] == "Shipping to Canada"
    assert resources.rewriter.calls == []  # type: ignore[union-attr]


def test_hundred_history_items_accepted_and_only_last_twelve_sent(
    seeded_client: TestClient, resources: AppResources
) -> None:
    history = [f"message {i}" for i in range(100)]
    ok(search(seeded_client, question=FOLLOW_UP, history=history))
    sent = resources.rewriter.calls[0][1]  # type: ignore[union-attr]
    assert sent == history[-12:]


def test_small_talk_returns_no_results(seeded_client: TestClient, resources: AppResources) -> None:
    resources.rewriter.needs_retrieval = False  # type: ignore[union-attr]
    data = ok(search(seeded_client, question="Thanks, that's all!", history=HISTORY))
    assert data["needs_retrieval"] is False
    assert data["results"] == []
    assert data["search_query"] == ""


def test_rewrite_failure_never_fails_the_search(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.rewriter.error = LLMTimeoutError()  # type: ignore[union-attr]
    data = ok(search(seeded_client, question=FOLLOW_UP, history=HISTORY))
    assert data["rewrite_fallback"] is True
    assert data["rewrite_source"] == "fallback"
    assert data["search_query"] == FOLLOW_UP
    assert data["results"]


def test_rewrite_disabled_provider_is_flagged(
    seeded_client: TestClient, resources: AppResources
) -> None:
    resources.rewriter = None
    resources.rewrite_provider = "none"
    data = ok(search(seeded_client, question=FOLLOW_UP, history=HISTORY))
    assert data["rewrite_source"] == "disabled"
    assert data["rewrite_fallback"] is True
    assert ok(seeded_client.get("/health"))["rewrite_provider"] == "none"


def test_strict_mode_surfaces_llm_errors_as_504_and_502(
    seeded_client: TestClient, resources: AppResources, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REWRITE_STRICT", "true")
    resources.rewriter.error = LLMTimeoutError()  # type: ignore[union-attr]
    err(search(seeded_client, question=FOLLOW_UP, history=HISTORY), 504, "LLM_TIMEOUT")
    resources.rewriter.error = LLMBudgetExhaustedError()  # type: ignore[union-attr]
    error = err(
        search(seeded_client, question=FOLLOW_UP, history=HISTORY), 502, "LLM_BUDGET_EXHAUSTED"
    )
    assert error["message"] == "LLM budget exhausted."


def test_search_runs_while_bm25_is_rebuilt(
    seeded_client: TestClient, resources: AppResources
) -> None:
    """A search during an index rebuild must never error (atomic swap)."""
    import threading

    chunks = resources.vector_store.get_all_chunks()
    stop = threading.Event()
    failures: list[BaseException] = []

    def rebuild_loop() -> None:
        try:
            while not stop.is_set():
                resources.keyword_index.rebuild(chunks)
                resources.keyword_index.rebuild(chunks[:2])
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=rebuild_loop)
    thread.start()
    try:
        for _ in range(25):
            response = search(
                seeded_client, question="waterproof shell jacket", query_mode="question_only"
            )
            assert response.status_code in (200, 409), (
                response.text
            )  # 409 only if momentarily empty
            assert response.status_code == 200
    finally:
        stop.set()
        thread.join(timeout=10)
    assert failures == []
