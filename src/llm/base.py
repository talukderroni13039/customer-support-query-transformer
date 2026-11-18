from typing import Protocol

from src.domains.support_search.schemas.chat_schemas import AnswerContextArticle, GeneratedAnswer
from src.domains.support_search.schemas.rewrite_schemas import RewrittenQuery


class QueryRewriter(Protocol):
    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery: ...


class AnswerGenerator(Protocol):
    async def answer(
        self, question: str, history: list[str], articles: list[AnswerContextArticle]
    ) -> GeneratedAnswer: ...
