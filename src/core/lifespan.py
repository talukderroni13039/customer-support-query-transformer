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
from src.utils.env import env_str
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
    index_write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def build_resources() -> AppResources:
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

    resources = AppResources(
        environment=env_str("APP_ENV", "dev"),
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
    )
    logger.info(
        "resources ready",
        extra={
            "embedding_model": embedder.model_name,
            "embedding_dimensions": embedder.dimensions,
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
