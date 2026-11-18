import pytest

from src.core.schemas.chunk import ChunkHit
from src.utils.rank_fusion import reciprocal_rank_fusion


def hit(chunk_id: str, score: float = 0.0) -> ChunkHit:
    return ChunkHit(chunk_id=chunk_id, article_id=chunk_id, title="t", text="x", score=score)


def test_hand_computed_two_list_example() -> None:
    """vector = [a, b, c], keyword = [b, d, a], k = 60.

    a: 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    b: 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    c: 1/63                              = 0.015873
    d: 1/62                              = 0.016129
    """
    vector = [hit("a", 0.91), hit("b", 0.80), hit("c", 0.10)]
    keyword = [hit("b", 14.2), hit("d", 9.0), hit("a", 1.5)]
    fused = reciprocal_rank_fusion([vector, keyword], k=60)
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]
    assert [round(h.score, 4) for h in fused] == [0.0325, 0.0323, 0.0161, 0.0159]


def test_item_in_both_lists_beats_item_in_one() -> None:
    fused = reciprocal_rank_fusion([[hit("solo"), hit("both")], [hit("other"), hit("both")]])
    assert fused[0].chunk_id == "both"
    assert fused[0].score == pytest.approx(2 / 62)


def test_raw_scores_are_ignored() -> None:
    low = reciprocal_rank_fusion([[hit("a", 0.001), hit("b", 0.0001)]])
    high = reciprocal_rank_fusion([[hit("a", 999.0), hit("b", 500.0)]])
    assert [h.score for h in low] == [h.score for h in high]


def test_works_with_one_empty_list_and_no_lists() -> None:
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], []])
    assert [h.chunk_id for h in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


def test_k_changes_scores_as_expected() -> None:
    lists = [[hit("a"), hit("b")], [hit("a")]]
    assert reciprocal_rank_fusion(lists, k=60)[0].score == pytest.approx(2 / 61)
    assert reciprocal_rank_fusion(lists, k=0)[0].score == pytest.approx(2.0)
    assert reciprocal_rank_fusion(lists, k=10)[1].score == pytest.approx(1 / 12)
    # Smaller k rewards top ranks more: the gap between rank 1 and rank 2 widens.
    gap_small_k = 1 / (1 + 1) - 1 / (1 + 2)
    gap_large_k = 1 / (60 + 1) - 1 / (60 + 2)
    assert gap_small_k > gap_large_k
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion(lists, k=-1)


def test_ties_are_deterministic_and_duplicates_count_once() -> None:
    fused = reciprocal_rank_fusion([[hit("b")], [hit("a")]])
    assert [h.chunk_id for h in fused] == ["a", "b"]  # equal score and rank → chunk_id order
    duplicated = reciprocal_rank_fusion([[hit("a"), hit("a"), hit("b")]])
    assert [round(h.score, 6) for h in duplicated] == [round(1 / 61, 6), round(1 / 62, 6)]


def test_inputs_are_not_mutated_and_fields_are_kept() -> None:
    original = ChunkHit(
        chunk_id="a", article_id="KB-1", title="T", category="c", heading="H", text="x", score=0.5
    )
    fused = reciprocal_rank_fusion([[original]])
    assert original.score == 0.5
    assert fused[0].model_dump(exclude={"score"}) == original.model_dump(exclude={"score"})
