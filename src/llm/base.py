from typing import Protocol

from src.domains.support_search.schemas.rewrite_schemas import RewrittenQuery


class QueryRewriter(Protocol):
    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery: ...
