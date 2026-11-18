from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]
Content = Annotated[str, StringConstraints(min_length=20, max_length=20_000)]
Category = Annotated[
    str, StringConstraints(strip_whitespace=True, max_length=50, pattern=r"^[a-z0-9-]+$")
]


class ArticleCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "title": "Returns for Outerwear",
                "content": "## Return window\nOuterwear can be returned within 60 days of "
                "delivery.\n\n## Condition requirements\nTags must be attached.",
                "category": "returns",
            }
        },
    )

    title: Title
    content: Content
    category: Category | None = None

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("content must contain at least 20 non-whitespace-padded characters")
        return value


class ArticleUpdate(ArticleCreate):
    """PUT replaces the whole article, so it has the same shape as create."""


class ArticleSummary(BaseModel):
    id: str
    title: str
    category: str | None = None
    chunk_count: int
    updated_at: str


class ArticleResult(BaseModel):
    id: str
    title: str
    category: str | None = None
    content: str
    chunk_count: int
    created_at: str
    updated_at: str


class ArticleDeleteResult(BaseModel):
    id: str
    deleted: Literal[True] = True
