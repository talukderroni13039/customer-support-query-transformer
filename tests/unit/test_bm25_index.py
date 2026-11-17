import threading

import pytest

from src.core.schemas.chunk import StoredChunk
from src.keyword_search.bm25_index import BM25Index


def make_chunk(
    chunk_id: str, text: str, *, title: str = "Product", heading: str = ""
) -> StoredChunk:
    article_id = chunk_id.split("::")[0]
    return StoredChunk(
        chunk_id=chunk_id,
        article_id=article_id,
        title=title,
        category="products",
        heading=heading,
        text=text,
        chunk_index=int(chunk_id.split("::")[1]),
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


CHUNKS = [
    make_chunk(
        "KB-101::0000", "Trailblazer Pro TB-PRO-22 is fully waterproof and machine washable."
    ),
    make_chunk(
        "KB-102::0000",
        "Trailblazer Lite TB-PRO-23 is waterproof; waterproof seams, machine washable.",
    ),
    make_chunk("KB-050::0000", "Our waterproof rating guide explains waterproof membranes."),
    make_chunk("KB-031::0000", "Shipping to Canada takes five business days."),
]


@pytest.fixture
def index() -> BM25Index:
    index = BM25Index()
    index.rebuild(CHUNKS)
    return index


def test_exact_code_query_ranks_matching_chunk_first(index: BM25Index) -> None:
    hits = index.search("Is TB-PRO-23 machine washable?", top_k=5)
    assert hits[0].chunk_id == "KB-102::0000"
    assert index.search("TB-PRO-22", top_k=5)[0].chunk_id == "KB-101::0000"


def test_tb_pro_22_query_does_not_rank_tb_pro_23_chunk_first(index: BM25Index) -> None:
    hits = index.search("TB-PRO-22", top_k=5)
    assert [h.chunk_id for h in hits] == ["KB-101::0000"]
    waterproof = index.search("Is TB-PRO-22 waterproof?", top_k=5)
    assert waterproof[0].chunk_id == "KB-101::0000"
    assert waterproof[0].chunk_id != "KB-102::0000"
    assert waterproof[0].score > waterproof[1].score


def test_hits_carry_chunk_fields_and_descending_positive_scores(index: BM25Index) -> None:
    hits = index.search("waterproof", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score > 0
    assert hits[0].title == "Product"
    assert hits[0].category == "products"
    assert index.search("waterproof", top_k=0) == []


def test_title_and_heading_are_searchable() -> None:
    index = BM25Index()
    index.rebuild(
        [
            make_chunk("KB-1::0000", "Within 60 days.", title="Returns", heading="Outerwear"),
            make_chunk("KB-2::0000", "Within 30 days.", title="Returns", heading="Footwear"),
        ]
    )
    assert index.search("footwear", top_k=5)[0].chunk_id == "KB-2::0000"


def test_matches_score_positive_on_a_tiny_corpus() -> None:
    index = BM25Index()
    index.rebuild([make_chunk("KB-1::0000", "gift cards never expire")])
    assert [h.chunk_id for h in index.search("gift cards", top_k=3)] == ["KB-1::0000"]
    index.rebuild(
        [
            make_chunk("KB-1::0000", "gift cards never expire"),
            make_chunk("KB-2::0000", "gift wrapping is free"),
        ]
    )
    assert index.search("gift", top_k=3)[0].score > 0
    assert len(index.search("gift", top_k=3)) == 2


def test_rebuild_after_delete_removes_the_chunk(index: BM25Index) -> None:
    index.rebuild([c for c in CHUNKS if c.article_id != "KB-101"])
    assert index.size() == 3
    assert index.search("TB-PRO-22", top_k=5) == []


def test_empty_index_and_empty_query_return_no_results(index: BM25Index) -> None:
    empty = BM25Index()
    assert empty.size() == 0
    assert empty.search("waterproof", top_k=5) == []
    empty.rebuild([])
    assert empty.search("waterproof", top_k=5) == []
    assert index.search("the of and", top_k=5) == []
    assert index.search("zzz-unknown-term", top_k=5) == []


def test_concurrent_search_during_rebuild_never_errors() -> None:
    index = BM25Index()
    index.rebuild(CHUNKS)
    errors: list[BaseException] = []
    stop = threading.Event()
    variants = [CHUNKS, CHUNKS[:2], [], CHUNKS[1:], CHUNKS * 3]

    def rebuild_loop() -> None:
        try:
            for i in range(300):
                index.rebuild(variants[i % len(variants)])
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def search_loop() -> None:
        try:
            while not stop.is_set():
                for hit in index.search("TB-PRO-22 waterproof shipping", top_k=10):
                    assert hit.score > 0
                index.size()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=rebuild_loop)]
    threads += [threading.Thread(target=search_loop) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert not any(thread.is_alive() for thread in threads)
