import pytest

from src.core.exceptions import VectorStoreError
from src.vectorstore.chroma_store import ChromaStore
from tests.fakes import FakeEmbedder, chunk_metadata, make_chroma_store

RETURN_TEXT = "## Return window\nJackets return within 60 days\n\n"


@pytest.fixture
def store() -> ChromaStore:
    store = make_chroma_store()
    embedder = FakeEmbedder()
    docs = {
        "KB-1::0000": (RETURN_TEXT, "KB-1", 0, "returns"),
        "KB-1::0001": ("## Condition\nTags attached", "KB-1", 1, "returns"),
        "KB-2::0000": ("Shipping to Canada takes 5 days", "KB-2", 0, "shipping"),
    }
    store.upsert(
        ids=list(docs),
        embeddings=embedder.embed_documents([d[0] for d in docs.values()]),
        documents=[d[0] for d in docs.values()],
        metadatas=[
            chunk_metadata(d[1], d[2], title=f"Title {d[1]}", category=d[3], heading="H")
            for d in docs.values()
        ],
    )
    return store


def test_upsert_count_and_exact_documents(store: ChromaStore) -> None:
    assert store.count() == 3
    chunks = store.get_article_chunks("KB-1")
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].text == RETURN_TEXT
    assert chunks[0].title == "Title KB-1"
    assert chunks[0].category == "returns"
    assert store.get_article_chunks("KB-404") == []


def test_upsert_replaces_same_id_and_empty_upsert_is_noop(store: ChromaStore) -> None:
    store.upsert([], [], [], [])
    store.upsert(
        ["KB-2::0000"],
        FakeEmbedder().embed_documents(["new text"]),
        ["new text"],
        [chunk_metadata("KB-2", 0)],
    )
    assert store.count() == 3
    assert store.get_article_chunks("KB-2")[0].text == "new text"


def test_get_all_chunks_sorted(store: ChromaStore) -> None:
    assert [c.chunk_id for c in store.get_all_chunks()] == [
        "KB-1::0000",
        "KB-1::0001",
        "KB-2::0000",
    ]


def test_delete_article_returns_removed_count(store: ChromaStore) -> None:
    assert store.delete_article("KB-1") == 2
    assert store.delete_article("KB-1") == 0
    assert store.count() == 1


def test_list_articles_with_category_filter(store: ChromaStore) -> None:
    summaries = store.list_articles()
    assert [(s.id, s.chunk_count) for s in summaries] == [("KB-1", 2), ("KB-2", 1)]
    assert [s.id for s in store.list_articles("shipping")] == ["KB-2"]
    assert store.list_articles("nope") == []


def test_query_returns_cosine_similarity_hits(store: ChromaStore) -> None:
    hits = store.query(FakeEmbedder().embed_query("shipping to canada"), top_k=10)
    assert len(hits) == 3
    assert hits[0].article_id == "KB-2"
    assert hits[0].score == pytest.approx(max(h.score for h in hits))
    assert -1.0001 <= hits[-1].score <= hits[0].score <= 1.0001
    assert hits[0].heading == "H"
    assert store.query(FakeEmbedder().embed_query("x"), top_k=0) == []


def test_empty_store_and_missing_category() -> None:
    store = make_chroma_store()
    assert store.count() == 0
    assert store.query(FakeEmbedder().embed_query("anything"), top_k=5) == []
    store.upsert(
        ["a::0000"], FakeEmbedder().embed_documents(["t"]), ["t"], [chunk_metadata("a", 0)]
    )
    assert store.get_all_chunks()[0].category is None
    assert store.list_articles()[0].category is None


def test_chroma_failures_become_vector_store_errors(store: ChromaStore) -> None:
    with pytest.raises(VectorStoreError):
        store.query([0.1, 0.2], top_k=3)  # wrong dimensionality

    class BrokenClient:
        def get_or_create_collection(self, **_: object) -> None:
            raise RuntimeError("down")

    with pytest.raises(VectorStoreError):
        ChromaStore(BrokenClient())
