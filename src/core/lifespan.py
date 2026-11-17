"""Startup: load models, open ChromaDB, build BM25 from ChromaDB. Everything is created once."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI

from src.embeddings.base import Embedder
from src.embeddings.bge_embedder import DEFAULT_EMBEDDING_MODEL, BgeEmbedder
from src.keyword_search.base import KeywordIndex
from src.keyword_search.bm25_index import BM25Index
from src.reranking.base import Reranker
from src.reranking.bge_reranker import DEFAULT_RERANKER_MODEL, BgeReranker
from src.utils.env import env_bool, env_str
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
    index_write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def build_resources(*, load_reranker: bool = True) -> AppResources:
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

    resources = AppResources(
        environment=env_str("APP_ENV", "dev"),
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        reranker=reranker,
    )
    logger.info(
        "resources ready",
        extra={
            "embedding_model": embedder.model_name,
            "embedding_dimensions": embedder.dimensions,
            "reranker_model": reranker.model_name if reranker else None,
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
