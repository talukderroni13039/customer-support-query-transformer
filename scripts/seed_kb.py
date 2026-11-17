"""Seed the knowledge base: load `data/kb/*.md` through ArticleService with fixed IDs.

Usage (PowerShell):  uv run python -m scripts.seed_kb
Idempotent: re-running replaces each article under the same ID.
"""

import asyncio
from pathlib import Path

import yaml

from src.core.lifespan import build_resources
from src.domains.support_search.chunking.article_chunker import DEFAULT_CHUNK_MAX_CHARS
from src.domains.support_search.schemas.article_schemas import ArticleCreate
from src.domains.support_search.services.article_service import ArticleService
from src.main import load_environment
from src.utils.env import env_int, env_str

FRONT_MATTER_DELIMITER = "---"


def parse_article_file(path: Path) -> tuple[str, ArticleCreate]:
    """Split YAML front matter (id, title, category) from the markdown body."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith(FRONT_MATTER_DELIMITER):
        raise ValueError(f"{path.name}: missing front matter")
    _, front_matter, body = text.split(FRONT_MATTER_DELIMITER, 2)
    meta = yaml.safe_load(front_matter) or {}
    missing = [key for key in ("id", "title") if not meta.get(key)]
    if missing:
        raise ValueError(f"{path.name}: front matter is missing {missing}")
    payload = ArticleCreate(
        title=str(meta["title"]),
        content=body.lstrip("\n"),
        category=str(meta["category"]) if meta.get("category") else None,
    )
    return str(meta["id"]), payload


def load_kb(kb_dir: Path) -> list[tuple[str, ArticleCreate]]:
    return [parse_article_file(path) for path in sorted(kb_dir.glob("*.md"))]


async def seed(service: ArticleService, kb_dir: Path) -> int:
    articles = load_kb(kb_dir)
    for article_id, payload in articles:
        await service.upsert(article_id, payload)
    return len(articles)


async def main() -> None:
    load_environment()
    resources = build_resources()
    service = ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
        chunk_max_chars=env_int("CHUNK_MAX_CHARS", DEFAULT_CHUNK_MAX_CHARS),
    )
    count = await seed(service, Path(env_str("KB_DIR", "./data/kb")))
    print(f"Seeded {count} articles ({resources.keyword_index.size()} chunks).")


if __name__ == "__main__":
    asyncio.run(main())
