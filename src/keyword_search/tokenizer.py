"""Text → BM25 tokens. The same function is used for chunks and for queries."""

import re

# A token is letters/digits, optionally joined by hyphens: `tb-pro-22` stays one token.
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

STOP_WORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "she",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "until",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, keep hyphenated codes whole, drop stop words.

    A hyphenated token containing a digit is a code (`tb-pro-22`, `ss-110`) and stays whole, so
    `TB-PRO-22` never matches `TB-PRO-23`. A hyphenated word without digits (`water-resistant`)
    is split into its parts so it also matches the unhyphenated spelling.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        is_plain_compound = "-" in raw and not any(ch.isdigit() for ch in raw)
        parts = raw.split("-") if is_plain_compound else [raw]
        tokens.extend(part for part in parts if part not in STOP_WORDS)
    return tokens
