"""Calibrate the chat relevance threshold on rerank scores.

Usage (PowerShell):  uv run python -m scripts.calibrate_threshold

Positives: the retrieval cases of data/eval/conversations.json (the score of the best chunk
from an expected article). Negatives: data/eval/negatives.json (the top score of questions
that are off-topic or not covered by any article). Rewrites come from the rewrite cache, so
only the first run after a prompt change calls Claude.

Prints, for a range of thresholds, how many correct answers would survive the gate and how
many negatives it would stop, and writes the raw scores to data/cache/threshold_scores.json.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb

from scripts.seed_kb import seed
from src.core.lifespan import build_rewriter
from src.domains.support_search.schemas.search_schemas import QueryMode, RetrievalMode
from src.domains.support_search.services.article_service import ArticleService
from src.domains.support_search.services.query_service import QueryService
from src.domains.support_search.services.retrieval_service import RetrievalService
from src.embeddings.bge_embedder import BgeEmbedder
from src.keyword_search.bm25_index import BM25Index
from src.main import PROJECT_ROOT, load_environment
from src.reranking.bge_reranker import BgeReranker
from src.vectorstore.chroma_store import ChromaStore

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "conversations.json"
NEGATIVES_PATH = PROJECT_ROOT / "data" / "eval" / "negatives.json"
RAW_PATH = PROJECT_ROOT / "data" / "cache" / "threshold_scores.json"
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
CHAT_MIN_ARTICLES = 3


def share(values: list[float], predicate: Any) -> str:
    if not values:
        return "  n/a"
    return f"{sum(1 for v in values if predicate(v)) / len(values):5.0%}"


async def main() -> None:
    load_environment()
    os.environ.setdefault("LLM_MAX_RETRIES", "1")
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    negatives = json.loads(NEGATIVES_PATH.read_text(encoding="utf-8"))

    embedder, reranker = BgeEmbedder(), BgeReranker()
    store = ChromaStore(chromadb.EphemeralClient(), f"calibrate-{uuid.uuid4().hex}")
    index = BM25Index()
    await seed(ArticleService(embedder, store, index, asyncio.Lock()), Path("data/kb"))
    rewriter, _ = build_rewriter()
    query_service = QueryService(rewriter)
    retrieval = RetrievalService(embedder, store, index, reranker)

    async def score(question: str, history: list[str], expected: list[str]) -> dict[str, Any]:
        plan = await query_service.build_query(question, history, QueryMode.REWRITE)
        row: dict[str, Any] = {
            "search_query": plan.search_query,
            "needs_retrieval": plan.needs_retrieval,
            "rewrite_fallback": plan.rewrite_fallback,
        }
        if not plan.needs_retrieval:
            return row
        outcome = await retrieval.retrieve(
            plan.search_query, RetrievalMode.HYBRID, rerank=True, min_articles=CHAT_MIN_ARTICLES
        )
        by_query = {c.hit.chunk_id: c.hit.score for c in outcome.chunks}
        # The same candidates scored against the customer's own words, for comparison.
        rescored = await asyncio.to_thread(
            reranker.rerank, question, [c.hit for c in outcome.chunks]
        )
        by_question = {h.chunk_id: h.score for h in rescored}
        article_of = {c.hit.chunk_id: c.hit.article_id for c in outcome.chunks}

        def best(scores: dict[str, float], only_expected: bool) -> float:
            picked = [
                s for cid, s in scores.items() if not only_expected or article_of[cid] in expected
            ]
            return max(picked, default=0.0)

        combined = {cid: max(by_query[cid], by_question[cid]) for cid in by_query}
        row.update(
            top_query=best(by_query, False),
            top_combined=best(combined, False),
            expected_query=best(by_query, True),
            expected_combined=best(combined, True),
        )
        return row

    positives: list[dict[str, Any]] = []
    for case in cases:
        if case["pattern"] == "small_talk":
            continue
        row = await score(case["question"], case["history"], case["expected_article_ids"])
        positives.append({"id": case["id"], "pattern": case["pattern"], **row})
    negative_rows: list[dict[str, Any]] = []
    for neg in negatives:
        row = await score(neg["question"], [], [])
        negative_rows.append({"id": neg["id"], "type": neg["type"], **row})

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(
        json.dumps({"positives": positives, "negatives": negative_rows}, indent=1),
        encoding="utf-8",
    )

    for label, pos_key, neg_key in (
        ("rewritten query only", "expected_query", "top_query"),
        ("max(rewritten query, original question)", "expected_combined", "top_combined"),
    ):
        pos = [p[pos_key] for p in positives]
        off = [n[neg_key] for n in negative_rows if n["type"] == "off_topic"]
        unc = [n[neg_key] for n in negative_rows if n["type"] == "not_covered"]
        print(f"\n=== scoring: {label} ===")
        print("threshold | correct answers kept | off-topic stopped | not-covered stopped")
        for t in THRESHOLDS:
            print(
                f"   {t:5.2f}  |        {share(pos, lambda v, t=t: v >= t)}         "
                f"|      {share(off, lambda v, t=t: v < t)}        |      "
                f"{share(unc, lambda v, t=t: v < t)}"
            )
        print("lowest correct-answer scores:", sorted(round(v, 3) for v in pos)[:8])
        print("highest off-topic scores:   ", sorted((round(v, 3) for v in off), reverse=True)[:5])
        print("highest not-covered scores: ", sorted((round(v, 3) for v in unc), reverse=True)[:8])

    print("\nlow-scoring correct cases (rewritten query only):")
    for p in sorted(positives, key=lambda p: p["expected_query"])[:8]:
        print(
            f"  {p['id']} {p['pattern']:<18} query={p['expected_query']:.3f} "
            f"combined={p['expected_combined']:.3f}  {p['search_query'][:70]!r}"
        )
    print(f"\nraw scores: {RAW_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
