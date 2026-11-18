from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from src.domains.support_search.schemas.search_schemas import (
    MAX_HISTORY_ITEMS,
    HistoryItem,
    Question,
)

MAX_ANSWER_CHARS = 4000


class AnswerKind(StrEnum):
    ANSWER = "answer"  # answered from the help articles
    NOT_FOUND = "not_found"  # a support question the articles do not cover
    OFF_TOPIC = "off_topic"  # not a question for this store's support
    SMALL_TALK = "small_talk"  # greeting, thanks, goodbye


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "How long do I have to return it?",
                "history": [
                    "Do you sell the Trailblazer jacket in waterproof?",
                    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.",
                ],
            }
        },
    )

    question: Question
    history: list[HistoryItem] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_ITEMS,
        description="Earlier messages as plain strings, oldest first. Only the last 12 are used.",
    )


class GeneratedAnswer(BaseModel):
    """The support assistant's reply to the customer's latest message."""

    kind: AnswerKind = Field(
        description=(
            "answer: the articles contain the answer. "
            "not_found: a genuine question for this store's support, but the articles do not "
            "contain the answer. "
            "off_topic: not about this store, its products, orders, or policies. "
            "small_talk: a greeting, thanks, or goodbye with no question."
        )
    )
    answer: str = Field(
        default="",
        max_length=MAX_ANSWER_CHARS,
        description=(
            "The reply to show the customer. Required for kind=answer and kind=small_talk. "
            "Leave empty for not_found and off_topic."
        ),
    )
    source_article_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the articles the answer is based on. Empty unless kind=answer.",
    )
    # Set by CachedAnswerGenerator, never by the model.
    from_cache: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    @field_validator("answer")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _answer_required(self) -> "GeneratedAnswer":
        if self.kind in (AnswerKind.ANSWER, AnswerKind.SMALL_TALK) and not self.answer:
            raise ValueError("answer must not be empty when kind is answer or small_talk")
        return self


class AnswerContextArticle(BaseModel):
    """One retrieved article handed to the answer generator."""

    article_id: str
    title: str
    content: str


class ChatSource(BaseModel):
    article_id: str
    title: str


class ChatResult(BaseModel):
    question: str
    answer: str
    kind: AnswerKind
    sources: list[ChatSource] = Field(default_factory=list)
    search_query: str = ""
    rewrite_fallback: bool = False
    # True when Claude could not write the answer and the best article is quoted instead.
    answer_fallback: bool = False
