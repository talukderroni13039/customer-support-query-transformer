"""Reciprocal Rank Fusion. Pure function: no I/O, no globals."""

from collections.abc import Sequence

from src.core.schemas.chunk import ChunkHit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ChunkHit]], k: int = DEFAULT_RRF_K
) -> list[ChunkHit]:
    """Fuse ranked lists by position only: `score(chunk) = Σ 1 / (k + rank)` (rank starts at 1).

    Raw scores are ignored, so lists on different scales (cosine, BM25) can be merged.
    Returns one hit per chunk with `score` set to the RRF score, best first. Ties are broken
    by the best rank the chunk reached in any list, then by `chunk_id`, so the output is
    deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, ChunkHit] = {}
    for hits in ranked_lists:
        seen_in_list: set[str] = set()
        rank = 0
        for hit in hits:
            if hit.chunk_id in seen_in_list:
                continue
            seen_in_list.add(hit.chunk_id)
            rank += 1
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best_rank[hit.chunk_id] = min(best_rank.get(hit.chunk_id, rank), rank)
            first_seen.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    return [first_seen[cid].with_score(scores[cid]) for cid in ordered]
