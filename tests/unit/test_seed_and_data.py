"""Seed script + integrity of the committed knowledge base and eval set."""

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.seed_kb import load_kb, parse_article_file, seed
from src.core.lifespan import AppResources
from src.domains.support_search.services.article_service import ArticleService

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "kb"
EVAL_PATH = ROOT / "data" / "eval" / "conversations.json"
PATTERNS = {
    "pronoun",
    "ellipsis",
    "correction",
    "multi_reference",
    "exact_code",
    "paraphrase",
    "already_standalone",
    "small_talk",
}


def test_parse_article_file(tmp_path: Path) -> None:
    path = tmp_path / "KB-900.md"
    path.write_text(
        "---\nid: KB-900\ntitle: Sample Article\ncategory: returns\n---\n\n"
        "## Heading\nBody text of the sample article.\n",
        encoding="utf-8",
    )
    article_id, payload = parse_article_file(path)
    assert article_id == "KB-900"
    assert payload.title == "Sample Article"
    assert payload.category == "returns"
    assert payload.content == "## Heading\nBody text of the sample article.\n"


def test_parse_article_file_rejects_bad_front_matter(tmp_path: Path) -> None:
    no_front_matter = tmp_path / "a.md"
    no_front_matter.write_text("just a body with no front matter at all", encoding="utf-8")
    with pytest.raises(ValueError, match="missing front matter"):
        parse_article_file(no_front_matter)
    no_id = tmp_path / "b.md"
    no_id.write_text("---\ntitle: T\n---\nbody text long enough to pass", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        parse_article_file(no_id)


def test_starter_kb_shape() -> None:
    articles = load_kb(KB_DIR)
    ids = [article_id for article_id, _ in articles]
    assert 30 <= len(articles) <= 40
    assert len(set(ids)) == len(ids)
    assert [path.stem for path in sorted(KB_DIR.glob("*.md"))] == ids
    categories = Counter(payload.category for _, payload in articles)
    assert 6 <= categories["products"] <= 10
    corpus = " ".join(payload.content for _, payload in articles)
    for near_duplicate_code in ("TB-PRO-22", "TB-PRO-23", "SS-110", "SS-111"):
        assert near_duplicate_code in corpus


def test_eval_set_shape_and_expected_ids_exist() -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    kb_ids = {article_id for article_id, _ in load_kb(KB_DIR)}
    assert 30 <= len(cases) <= 35
    assert len({case["id"] for case in cases}) == len(cases)
    per_pattern = Counter(case["pattern"] for case in cases)
    assert set(per_pattern) == PATTERNS
    assert all(3 <= count <= 5 for count in per_pattern.values())
    for case in cases:
        assert 1 <= len(case["question"]) <= 2000
        assert all(isinstance(item, str) for item in case["history"])
        assert set(case["expected_article_ids"]) <= kb_ids
        if case["pattern"] == "small_talk":
            assert case["expected_article_ids"] == []
            assert case["history"], (
                "small talk is only detected by the rewriter, which needs history"
            )
        else:
            assert case["expected_article_ids"]
    # The rewrite cache must cover the benchmark with at most 35 Claude calls.
    assert sum(1 for case in cases if case["history"]) <= 35


async def test_seed_loads_everything_and_is_idempotent(resources: AppResources) -> None:
    service = ArticleService(
        embedder=resources.embedder,
        vector_store=resources.vector_store,
        keyword_index=resources.keyword_index,
        write_lock=resources.index_write_lock,
    )
    count = await seed(service, KB_DIR)
    chunks_after_first = resources.vector_store.count()
    assert count == len(list(KB_DIR.glob("*.md")))
    assert len(await service.list_articles()) == count
    assert resources.keyword_index.size() == chunks_after_first

    assert await seed(service, KB_DIR) == count
    assert resources.vector_store.count() == chunks_after_first

    source = (KB_DIR / "KB-012.md").read_text(encoding="utf-8").split("---", 2)[2].lstrip("\n")
    assert (await service.get("KB-012")).content == source
    assert resources.keyword_index.search("TB-PRO-23", top_k=1)[0].article_id == "KB-102"
