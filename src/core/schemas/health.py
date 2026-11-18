from typing import Literal

from pydantic import BaseModel


class HealthResult(BaseModel):
    status: Literal["ok", "degraded"]
    environment: str
    embedding_model_loaded: bool
    reranker_loaded: bool
    vector_store: Literal["ok", "error", "unavailable"]
    keyword_index_chunks: int
    rewrite_provider: str
