from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from src.core.schemas.chunk import chunk_body
from src.domains.support_search.schemas.search_schemas import (
    MAX_HISTORY_ITEMS,
    HistoryItem,
    Question,
)
from src.utils.grounding import check_grounding

MAX_ANSWER_CHARS = 4000
MAX_QUOTE_CHARS = 600


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
        description=(
            "Earlier messages, oldest first: plain strings, or {role, text} with role "
            "'customer' or 'agent'. Only the last 12 are used."
        ),
    )


class ContextChunk(BaseModel):
    """One retrieved chunk handed to the answer generator under a short label (C1, C2, ...)."""

    label: str
    chunk_id: str
    article_id: str
    title: str
    heading: str = ""
    text: str
    score: float = 0.0

    @property
    def source(self) -> str:
        return f"{self.title} / {self.heading}" if self.heading else self.title

    @property
    def body(self) -> str:
        return chunk_body(self.heading, self.text)

    @property
    def checkable_text(self) -> str:
        """Everything a citation of this chunk may legitimately quote or take numbers from."""
        return f"{self.title} {self.heading} {self.text}"


class Citation(BaseModel):
    chunk_id: str = Field(description="The label of the chunk the fact comes from, e.g. C1.")
    quote: str = Field(
        max_length=MAX_QUOTE_CHARS,
        description="The exact words copied from that chunk that support the fact.",
    )

    @field_validator("chunk_id")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return value.strip().strip("[]").upper()


class GeneratedAnswer(BaseModel):
    """The support assistant's reply to the customer's latest message."""

    kind: AnswerKind = Field(
        description=(
            "answer: the chunks contain the answer. "
            "not_found: a genuine question for this store's support, but the chunks do not "
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
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "One entry per fact in the answer: the chunk it comes from and the exact quote. "
            "Required for kind=answer; empty for every other kind."
        ),
    )
    # Set by CachedAnswerGenerator, never by the model.
    from_cache: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    @field_validator("answer")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> "GeneratedAnswer":
        if self.kind in (AnswerKind.ANSWER, AnswerKind.SMALL_TALK) and not self.answer:
            raise ValueError("answer must not be empty when kind is answer or small_talk")
        if self.kind is not AnswerKind.ANSWER:
            return self
        if not self.citations:
            raise ValueError("kind=answer needs at least one citation with a quote")

        # Grounding check. The chunks are supplied as validation context by the generator,
        # so a failure here goes back to the model once as a correction request.
        context = info.context or {}
        chunks: dict[str, str] | None = context.get("chunks")
        if chunks is None:
            return self
        problems = check_grounding(
            self.answer,
            [(c.chunk_id, c.quote) for c in self.citations],
            chunks,
            context.get("question_text", ""),
            context.get("history_text", ""),
            context.get("titles_text", ""),
        )
        if problems:
            raise ValueError(
                "The answer is not grounded in the provided chunks: "
                + "; ".join(problems)
                + ". Fix exactly these problems and answer again: cite the chunk each fact "
                "comes from with a quote copied word for word, and remove any statement no chunk "
                "supports. Keep answering the question if the chunks do answer it."
            )
        return self


class ChatSource(BaseModel):
    article_id: str
    title: str


class ChatCitation(BaseModel):
    article_id: str
    title: str
    heading: str = ""
    quote: str


class ChatResult(BaseModel):
    question: str
    answer: str
    kind: AnswerKind
    sources: list[ChatSource] = Field(default_factory=list)
    citations: list[ChatCitation] = Field(default_factory=list)
    search_query: str = ""
    rewrite_fallback: bool = False
    # True when Claude was unavailable and the best chunk is quoted instead.
    answer_fallback: bool = False
    # True when the generated answer failed the grounding check twice and was withheld.
    grounding_failed: bool = False
    # What the output guardrail changed, e.g. ["masked_email"]. Empty when nothing was changed.
    guardrail_actions: list[str] = Field(default_factory=list)
