import pytest

from src.core.schemas.chunk import compose_index_text
from src.domains.support_search.chunking.article_chunker import chunk_article

ARTICLE = (
    "Intro paragraph before any heading.\n\n"
    "## Return window\nOuterwear can be returned within 60 days.\n\n"
    "## Condition requirements\nTags must be attached.\n\nItems must be unworn.\n"
)


def _joined(content: str, max_chars: int = 1200) -> str:
    return "".join(chunk.text for chunk in chunk_article(content, max_chars))


def test_one_chunk_per_heading_section_with_preamble() -> None:
    chunks = chunk_article(ARTICLE)
    assert [c.heading for c in chunks] == ["", "Return window", "Condition requirements"]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert chunks[1].text.startswith("## Return window\n")


@pytest.mark.parametrize("max_chars", [1200, 60, 25, 7])
def test_chunks_rebuild_content_exactly(max_chars: int) -> None:
    assert _joined(ARTICLE, max_chars) == ARTICLE
    odd = "\n\n  leading blank\r\n\r\n## H\n\n\nbody   \n\n\n"
    assert _joined(odd, max_chars) == odd


def test_long_section_splits_by_paragraph_and_keeps_heading() -> None:
    content = "## Shipping\n" + "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(6))
    chunks = chunk_article(content, max_chars=200)
    assert len(chunks) > 1
    assert {c.heading for c in chunks} == {"Shipping"}
    assert all(len(c.text) <= 200 for c in chunks)


def test_no_headings_splits_by_paragraphs_up_to_max() -> None:
    content = "\n\n".join("word " * 30 for _ in range(5))
    chunks = chunk_article(content, max_chars=320)
    assert len(chunks) >= 3
    assert all(c.heading == "" for c in chunks)
    assert all(len(c.text) <= 320 for c in chunks)


def test_single_oversized_paragraph_is_hard_split_on_whitespace() -> None:
    content = "alpha " * 100
    chunks = chunk_article(content, max_chars=50)
    assert all(len(c.text) <= 50 for c in chunks)
    assert all(c.text.endswith(" ") for c in chunks)
    unbroken = "z" * 120
    assert [len(c.text) for c in chunk_article(unbroken, max_chars=50)] == [50, 50, 20]


def test_whitespace_only_pieces_are_merged_and_never_stand_alone() -> None:
    chunks = chunk_article("   \n\n## A\nbody a\n\n   \n\n## B\nbody b", max_chars=12)
    assert all(c.text.strip() for c in chunks)
    assert len(chunk_article("    ")) == 1


def test_invalid_max_chars_rejected() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_article("text", max_chars=0)


def test_compose_index_text_drops_duplicate_heading_line() -> None:
    text = "## Return window\nWithin 60 days.\n\n"
    assert compose_index_text("Returns", "Return window", text) == (
        "Returns — Return window: Within 60 days."
    )
    assert compose_index_text("Returns", "", " Plain body ") == "Returns: Plain body"
    assert compose_index_text("T", "Other", "## Different\nx") == "T — Other: ## Different\nx"
