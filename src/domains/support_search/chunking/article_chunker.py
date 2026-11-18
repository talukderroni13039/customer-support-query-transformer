"""Article content → chunks.

Rules:
- Markdown headings start a new section; one chunk per section.
- A section longer than `max_chars` is split by paragraph (then by whitespace if one paragraph
  is still too long). Content without headings is therefore split by paragraphs.
- Chunks are raw, contiguous slices: `"".join(chunk.text) == content`, so an article can be
  rebuilt exactly from its stored chunks.
"""

import re
from dataclasses import dataclass

DEFAULT_CHUNK_MAX_CHARS = 1200

_HEADING_LINE = re.compile(r"^#{1,6}[ \t]+\S.*$", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


@dataclass(frozen=True)
class ArticleChunk:
    index: int
    heading: str
    text: str


def chunk_article(content: str, max_chars: int = DEFAULT_CHUNK_MAX_CHARS) -> list[ArticleChunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    pieces: list[tuple[str, str]] = []
    for heading, section in _split_sections(content):
        for part in _split_long(section, max_chars):
            pieces.append((heading, part))

    merged = _merge_blank_pieces(pieces)
    return [ArticleChunk(index=i, heading=h, text=t) for i, (h, t) in enumerate(merged)]


def _split_sections(content: str) -> list[tuple[str, str]]:
    starts = [m.start() for m in _HEADING_LINE.finditer(content)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    sections: list[tuple[str, str]] = []
    for begin, end in zip(starts, [*starts[1:], len(content)], strict=True):
        section = content[begin:end]
        if not section:
            continue
        match = _HEADING_LINE.match(section)
        heading = match.group(0).lstrip("#").strip() if match else ""
        sections.append((heading, section))
    return sections


def _split_long(section: str, max_chars: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    # Paragraphs, each keeping its trailing separator so slices stay contiguous.
    paragraphs: list[str] = []
    cursor = 0
    for brk in _PARAGRAPH_BREAK.finditer(section):
        paragraphs.append(section[cursor : brk.end()])
        cursor = brk.end()
    if cursor < len(section):
        paragraphs.append(section[cursor:])

    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > max_chars:
            parts.append(current)
            current = ""
        if len(paragraph) > max_chars:
            hard = _hard_split(paragraph, max_chars)
            parts.extend(hard[:-1])
            current = hard[-1]
        else:
            current += paragraph
    if current:
        parts.append(current)
    return parts


def _hard_split(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    while len(text) > max_chars:
        cut = max_chars
        for i in range(max_chars, max_chars // 2, -1):
            if text[i - 1].isspace():
                cut = i
                break
        parts.append(text[:cut])
        text = text[cut:]
    if text:
        parts.append(text)
    return parts


def _merge_blank_pieces(pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Whitespace-only pieces carry no meaning: glue them onto a neighbour."""
    merged: list[tuple[str, str]] = []
    pending = ""
    for heading, text in pieces:
        if not text.strip():
            if merged:
                last_heading, last_text = merged[-1]
                merged[-1] = (last_heading, last_text + text)
            else:
                pending += text
            continue
        merged.append((heading, pending + text))
        pending = ""
    if pending and not merged:
        merged.append(("", pending))
    return merged
