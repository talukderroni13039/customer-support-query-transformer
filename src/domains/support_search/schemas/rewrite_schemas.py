from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

MAX_REWRITE_CHARS = 500


class RewrittenQuery(BaseModel):
    """The latest customer message rewritten as one standalone knowledge-base search query."""

    needs_retrieval: bool = Field(
        description=(
            "false only when the latest customer message is small talk with no information "
            "need (thanks, greetings, goodbyes, acknowledgements); otherwise true."
        )
    )
    standalone_query: str = Field(
        default="",
        max_length=MAX_REWRITE_CHARS,
        description=(
            "One self-contained knowledge-base search query for the latest customer message. "
            "Product names, SKUs, and order numbers copied exactly as written. "
            "Empty string when needs_retrieval is false."
        ),
    )
    # Set by CachedRewriter, never by the model: hidden from the JSON schema sent to Claude.
    from_cache: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    @field_validator("standalone_query")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def _query_required_when_retrieving(self) -> "RewrittenQuery":
        if self.needs_retrieval and not self.standalone_query:
            raise ValueError("standalone_query must not be empty when needs_retrieval is true")
        return self
