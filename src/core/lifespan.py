"""Startup: load models, open ChromaDB, build BM25 from ChromaDB. Everything is created once."""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI

from src.domains.support_search.prompts.answer_prompt import ANSWER_PROMPT_VERSION
from src.domains.support_search.prompts.rewrite_prompt import PROMPT_VERSION
from src.embeddings.base import Embedder
from src.embeddings.bge_embedder import DEFAULT_EMBEDDING_MODEL, BgeEmbedder
from src.keyword_search.base import KeywordIndex
from src.keyword_search.bm25_index import BM25Index
from src.llm.anthropic_answerer import (
    DEFAULT_ANSWER_MAX_RETRIES,
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_TIMEOUT_SECONDS,
    AnthropicAnswerGenerator,
)
from src.llm.anthropic_rewriter import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    AnthropicRewriter,
)
from src.llm.base import AnswerGenerator, QueryRewriter
from src.llm.cached_answerer import DEFAULT_ANSWER_CACHE_PATH, CachedAnswerGenerator
from src.llm.cached_rewriter import DEFAULT_REWRITE_CACHE_PATH, CachedRewriter
from src.llm.call_budget import DEFAULT_BUDGET_PATH, DEFAULT_LLM_CALL_BUDGET, LlmCallBudget
from src.reranking.base import Reranker
from src.reranking.bge_reranker import DEFAULT_RERANKER_MODEL, BgeReranker
from src.utils.env import env_bool, env_float, env_int, env_str
from src.vectorstore.base import VectorStore
from src.vectorstore.chroma_store import DEFAULT_COLLECTION, ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class AppResources:
    """Long-lived infrastructure objects, stored on `app.state.resources`."""

    environment: str
    embedder: Embedder
    vector_store: VectorStore
    keyword_index: KeywordIndex
    reranker: Reranker | None = None  # None when RERANK_ENABLED=false
    rewriter: QueryRewriter | None = None  # None when REWRITE_PROVIDER=none or no API key
    rewrite_provider: str = "none"
    answer_generator: AnswerGenerator | None = None  # None without a provider or key
    index_write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _llm_api_key() -> str | None:
    """The Anthropic key, or None when the provider is disabled or the key is missing."""
    provider = env_str("REWRITE_PROVIDER", "anthropic").lower()
    if provider == "none":
        return None
    if provider != "anthropic":
        raise ValueError("REWRITE_PROVIDER must be 'anthropic' or 'none'")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY is not set: rewrite and answers fall back")
        return None
    return api_key


def _call_budget() -> LlmCallBudget:
    return LlmCallBudget(
        path=Path(env_str("LLM_BUDGET_PATH", DEFAULT_BUDGET_PATH)),
        limit=env_int("LLM_CALL_BUDGET", DEFAULT_LLM_CALL_BUDGET),
    )


def build_rewriter() -> tuple[QueryRewriter | None, str]:
    """Create the cached Claude rewriter once. Returns `(rewriter, provider_name)`."""
    api_key = _llm_api_key()
    if api_key is None:
        return None, "none"

    model = env_str("LLM_MODEL", DEFAULT_LLM_MODEL)
    budget = _call_budget()
    rewriter = AnthropicRewriter(
        api_key=api_key,
        budget=budget,
        model=model,
        max_tokens=env_int("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),
        max_retries=env_int("LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES),
        timeout_seconds=env_float("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS),
    )
    cached = CachedRewriter(
        inner=rewriter,
        cache_path=Path(env_str("REWRITE_CACHE_PATH", DEFAULT_REWRITE_CACHE_PATH)),
        namespace=f"{model}:{PROMPT_VERSION}",
    )
    return cached, "anthropic"


def build_answer_generator() -> AnswerGenerator | None:
    """Create the cached Claude answer generator once (same provider, key, and budget)."""
    api_key = _llm_api_key()
    if api_key is None:
        return None
    model = env_str("LLM_MODEL", DEFAULT_LLM_MODEL)
    generator = AnthropicAnswerGenerator(
        api_key=api_key,
        budget=_call_budget(),
        model=model,
        max_tokens=env_int("LLM_ANSWER_MAX_TOKENS", DEFAULT_ANSWER_MAX_TOKENS),
        max_retries=env_int("LLM_ANSWER_MAX_RETRIES", DEFAULT_ANSWER_MAX_RETRIES),
        timeout_seconds=env_float("LLM_ANSWER_TIMEOUT_SECONDS", DEFAULT_ANSWER_TIMEOUT_SECONDS),
    )
    return CachedAnswerGenerator(
        inner=generator,
        cache_path=Path(env_str("ANSWER_CACHE_PATH", DEFAULT_ANSWER_CACHE_PATH)),
        namespace=f"{model}:{ANSWER_PROMPT_VERSION}",
    )


def build_resources(*, load_reranker: bool = True, load_rewriter: bool = True) -> AppResources:
    """Blocking (model loading, disk I/O): run in a worker thread or a script."""
    started = time.perf_counter()
    embedder = BgeEmbedder(
        model_name=env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        device=env_str("EMBEDDING_DEVICE", "cpu"),
    )
    vector_store = ChromaStore.persistent(
        path=env_str("CHROMA_PATH", "./data/chroma"),
        collection_name=env_str("CHROMA_COLLECTION", DEFAULT_COLLECTION),
    )
    keyword_index = BM25Index()
    keyword_index.rebuild(vector_store.get_all_chunks())

    reranker: Reranker | None = None
    if load_reranker and env_bool("RERANK_ENABLED", True):
        reranker = BgeReranker(
            model_name=env_str("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            device=env_str("EMBEDDING_DEVICE", "cpu"),
        )

    rewriter, rewrite_provider = build_rewriter() if load_rewriter else (None, "none")
    answer_generator = build_answer_generator() if load_rewriter else None

    resources = AppResources(
        environment=env_str("APP_ENV", "dev"),
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        reranker=reranker,
        rewriter=rewriter,
        rewrite_provider=rewrite_provider,
        answer_generator=answer_generator,
    )
    logger.info(
        "resources ready",
        extra={
            "embedding_model": embedder.model_name,
            "embedding_dimensions": embedder.dimensions,
            "reranker_model": reranker.model_name if reranker else None,
            "rewrite_provider": rewrite_provider,
            "keyword_index_chunks": keyword_index.size(),
            "startup_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return resources


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.resources = await asyncio.to_thread(build_resources)
    try:
        yield
    finally:
        app.state.resources = None
