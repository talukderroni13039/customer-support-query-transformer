from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_TEXT_CHARS = 2000
MAX_HISTORY_ITEMS = 100

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)
]
HistoryItem = Annotated[str, StringConstraints(max_length=MAX_TEXT_CHARS)]


class QueryMode(StrEnum):
    REWRITE = "rewrite"
    QUESTION_ONLY = "question_only"
    FULL_HISTORY = "full_history"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


class ScoreType(StrEnum):
    COSINE = "cosine"
    BM25 = "bm25"
    RRF = "rrf"
    RERANK = "rerank"


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "How long do I have to return it?",
                "history": [
                    "Do you sell the Trailblazer jacket in waterproof?",
                    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.",
                ],
                "top_k": 3,
                "query_mode": "rewrite",
                "retrieval": "hybrid",
                "rerank": True,
                "include_debug": True,
            }
        },
    )

    question: Question
    history: list[HistoryItem] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_ITEMS,
        description="Earlier messages as plain strings, oldest first. Only the last 12 are used.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Articles to return. Defaults to DEFAULT_TOP_K (3)."
    )
    query_mode: QueryMode = QueryMode.REWRITE
    retrieval: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    include_debug: bool = False


class StageDebug(BaseModel):
    """Per-result stage details for the best chunk of the article (ranks start at 1)."""

    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievedArticle(BaseModel):
    article_id: str
    title: str
    category: str | None = None
    score: float
    snippet: str
    debug: StageDebug | None = None


class SupportSearchResult(BaseModel):
    question: str
    search_query: str
    query_mode: QueryMode
    retrieval: RetrievalMode
    rerank: bool
    needs_retrieval: bool = True
    rewrite_source: str = "none"
    rewrite_fallback: bool = False
    rerank_fallback: bool = False
    score_type: ScoreType
    results: list[RetrievedArticle]
