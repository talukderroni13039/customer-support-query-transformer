"""Chunk shapes shared by every pipeline stage and every infrastructure Protocol."""

from pydantic import BaseModel, ConfigDict


def chunk_body(heading: str, text: str) -> str:
    """Raw chunk text without its own heading line, stripped."""
    body = text.strip()
    if heading and body.startswith("#"):
        first_line, _, rest = body.partition("\n")
        if first_line.lstrip("#").strip() == heading:
            body = rest.strip()
    return body


def compose_index_text(title: str, heading: str, text: str) -> str:
    """Text used for embedding, BM25, and reranking: `"{title} — {heading}: {chunk}"`.

    The stored chunk keeps its raw text (including the heading line) so the article can be
    rebuilt exactly; here the heading line is dropped from the body to avoid repeating it.
    """
    body = chunk_body(heading, text)
    prefix = f"{title} — {heading}" if heading else title
    return f"{prefix}: {body}"


class StoredChunk(BaseModel):
    """A chunk as persisted in the vector store (the article source of truth)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    article_id: str
    title: str
    category: str | None = None
    heading: str = ""
    text: str
    chunk_index: int
    created_at: str
    updated_at: str

    @property
    def index_text(self) -> str:
        return compose_index_text(self.title, self.heading, self.text)


class ChunkHit(BaseModel):
    """One retrieved chunk with the score of the stage that produced it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    article_id: str
    title: str
    category: str | None = None
    heading: str = ""
    text: str
    score: float

    @property
    def index_text(self) -> str:
        return compose_index_text(self.title, self.heading, self.text)

    def with_score(self, score: float) -> "ChunkHit":
        return self.model_copy(update={"score": score})
