from typing import Any

import pytest
from fastapi.testclient import TestClient

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
