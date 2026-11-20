"""Application errors. Raise these from services; the global handler renders them."""

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: Any | None = None) -> None:
        self.message = message or type(self).message
        self.details = details
        super().__init__(self.message)


class ArticleNotFoundError(AppError):
    status_code = 404
    code = "ARTICLE_NOT_FOUND"
    message = "Article not found."


class KnowledgeBaseEmptyError(AppError):
    status_code = 409
    code = "KB_EMPTY"
    message = "The knowledge base is empty. Add or seed articles before searching."


class RerankerUnavailableError(AppError):
    status_code = 409
    code = "RERANKER_DISABLED"
    message = "Reranking is disabled on this server (RERANK_ENABLED=false). Send rerank=false."


class ModelNotReadyError(AppError):
    status_code = 503
    code = "MODEL_NOT_READY"
    message = "Models are not loaded yet."


class VectorStoreError(AppError):
    status_code = 503
    code = "VECTOR_STORE_ERROR"
    message = "The vector store is unavailable."


class LLMError(AppError):
    status_code = 502
    code = "LLM_ERROR"
    message = "The query rewrite provider returned an error."


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "LLM_TIMEOUT"
    message = "The query rewrite provider timed out."


class LLMBudgetExhaustedError(LLMError):
    status_code = 502
    code = "LLM_BUDGET_EXHAUSTED"
    message = "LLM budget exhausted."


class LLMOutputInvalidError(LLMError):
    """The model's output failed validation (schema or grounding check) on every attempt."""

    status_code = 502
    code = "LLM_OUTPUT_INVALID"
    message = "The LLM output failed validation."
