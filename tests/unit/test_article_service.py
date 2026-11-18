import asyncio

import pytest

from src.core.exceptions import ArticleNotFoundError
from src.core.lifespan import AppResources
from src.domains.support_search.schemas import ArticleCreate
from src.domains.support_search.services.article_service import ArticleService

CONTENT = (
    "## Return window\nOuterwear such as the TB-PRO-22 can be returned within 60 days.\n\n"
    "## Condition requirements\nTags must be attached and the item unworn.\n"
)


@pytest.fixture
def service(resources: AppResources) -> ArticleService:
    return ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
    )


def payload(title: str = "Returns for Outerwear", content: str = CONTENT) -> ArticleCreate:
    return ArticleCreate(title=title, content=content, category="returns")


async def test_create_returns_id_and_chunk_count(service: ArticleService) -> None:
    result = await service.create(payload())
    assert result.id.startswith("KB-")
    assert len(result.id) == 9
    assert result.chunk_count == 2
    assert result.created_at == result.updated_at
    assert result.created_at.endswith("Z")


async def test_get_rebuilds_exact_content(service: ArticleService) -> None:
    odd_content = "Intro line.\n\n\n## A\nBody of section A.  \n\n## B\n\nBody of section B.\n\n"
    created = await service.create(payload(content=odd_content))
    fetched = await service.get(created.id)
    assert fetched.content == odd_content
    assert fetched.title == "Returns for Outerwear"
    assert fetched.category == "returns"


async def test_embeds_title_heading_and_chunk(
    service: ArticleService, resources: AppResources
) -> None:
    await service.create(payload())
    embedded = resources.embedder.document_calls[-1]  # type: ignore[attr-defined]
    assert embedded[0].startswith("Returns for Outerwear — Return window: Outerwear such as")
    assert "## Return window" not in embedded[0]


async def test_every_change_rebuilds_bm25(service: ArticleService, resources: AppResources) -> None:
    index = resources.keyword_index
    assert index.size() == 0

    created = await service.create(payload())
    assert index.size() == 2
    assert index.search("TB-PRO-22", top_k=3)[0].article_id == created.id  # immediately findable

    await service.update(
        created.id, payload(content="## Window\nFootwear CH-310 returns within 30 days only.")
    )
    assert index.size() == 1
    assert index.search("TB-PRO-22", top_k=3) == []
    assert index.search("CH-310", top_k=3)[0].article_id == created.id

    await service.delete(created.id)
    assert index.size() == 0
    assert index.search("CH-310", top_k=3) == []


async def test_update_replaces_chunks_and_keeps_created_at(
    service: ArticleService, resources: AppResources
) -> None:
    created = await service.create(payload())
    updated = await service.update(
        created.id, payload(title="New title", content="Only one short paragraph now, no headings.")
    )
    assert updated.id == created.id
    assert updated.chunk_count == 1
    assert updated.title == "New title"
    assert updated.created_at == created.created_at
    assert resources.vector_store.count() == 1
    assert (await service.get(created.id)).content == "Only one short paragraph now, no headings."


async def test_delete_removes_chunks(service: ArticleService, resources: AppResources) -> None:
    created = await service.create(payload())
    deleted = await service.delete(created.id)
    assert deleted.id == created.id
    assert deleted.deleted is True
    assert resources.vector_store.count() == 0
    with pytest.raises(ArticleNotFoundError):
        await service.get(created.id)


async def test_unknown_article_raises_not_found(service: ArticleService) -> None:
    for call in (
        service.get("KB-404"),
        service.update("KB-404", payload()),
        service.delete("KB-404"),
    ):
        with pytest.raises(ArticleNotFoundError):
            await call


async def test_upsert_uses_fixed_id_and_is_idempotent(
    service: ArticleService, resources: AppResources
) -> None:
    first = await service.upsert("KB-012", payload())
    second = await service.upsert("KB-012", payload(title="Returns v2"))
    assert first.id == second.id == "KB-012"
    assert second.created_at == first.created_at
    assert resources.vector_store.count() == 2
    assert [a.title for a in await service.list_articles()] == ["Returns v2"]


async def test_list_filters_by_category(service: ArticleService) -> None:
    await service.upsert("KB-1", payload())
    await service.upsert(
        "KB-2", ArticleCreate(title="Shipping", content="Shipping to Canada takes five days.")
    )
    assert [a.id for a in await service.list_articles()] == ["KB-1", "KB-2"]
    assert [a.id for a in await service.list_articles("returns")] == ["KB-1"]
    assert (await service.list_articles())[1].category is None


async def test_failed_embedding_leaves_existing_article_untouched(
    service: ArticleService, resources: AppResources, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = await service.create(payload())

    def broken(_: list[str]) -> list[list[float]]:
        raise RuntimeError("model crashed")

    monkeypatch.setattr(resources.embedder, "embed_documents", broken)
    with pytest.raises(RuntimeError):
        await service.update(created.id, payload(content="Replacement content that never lands."))
    assert (await service.get(created.id)).content == CONTENT
    assert resources.keyword_index.size() == 2


async def test_concurrent_writes_leave_indexes_consistent(
    service: ArticleService, resources: AppResources
) -> None:
    await asyncio.gather(
        *(
            service.upsert(f"KB-{i:03d}", payload(content=f"Article number {i} about item XX-{i}."))
            for i in range(8)
        )
    )
    assert resources.vector_store.count() == 8
    assert resources.keyword_index.size() == 8
