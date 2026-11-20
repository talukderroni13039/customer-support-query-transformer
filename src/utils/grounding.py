"""Grounding check for generated answers. Pure functions: no I/O, no model calls.

An answer is grounded when every citation points at a chunk that was really provided, every
quote can be found in that chunk, and every number in the answer appears in a cited chunk
(or in the customer's own words). The check is deliberately lenient about wording: it is
there to catch invented facts, not harmless differences in punctuation or spacing.
"""

import re

QUOTE_MIN_TOKEN_COVERAGE = 0.8
QUOTE_MIN_TOKENS_FOR_FUZZY = 4

_NON_WORD = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalize(text: str) -> str:
    """Lowercase and reduce every run of punctuation or whitespace to one space."""
    return _NON_WORD.sub(" ", text.lower()).strip()


def quote_supported(quote: str, chunk_text: str) -> bool:
    """True when the quote appears in the chunk, ignoring case, punctuation, and spacing.

    A quote of four or more words also passes when at least 80% of its words occur in the
    chunk: models sometimes drop or swap a small word when copying.
    """
    wanted, source = normalize(quote), normalize(chunk_text)
    if not wanted:
        return False
    if wanted in source:
        return True
    tokens = wanted.split()
    if len(tokens) < QUOTE_MIN_TOKENS_FOR_FUZZY:
        return False
    source_tokens = set(source.split())
    covered = sum(1 for token in tokens if token in source_tokens)
    return covered / len(tokens) >= QUOTE_MIN_TOKEN_COVERAGE


def extract_numbers(text: str) -> set[str]:
    """Numbers as written, with thousands separators removed (`20,000` → `20000`).

    Single digits are ignored: they are usually list markers or counts like "a 2-person
    tent" restated loosely, and checking them causes false alarms.
    """
    numbers: set[str] = set()
    for match in _NUMBER.findall(text):
        value = match.replace(",", "").rstrip(".")
        if len(value.replace(".", "")) > 1:
            numbers.add(value)
    return numbers


def check_grounding(
    answer: str,
    citations: list[tuple[str, str]],
    chunks: dict[str, str],
    customer_text: str = "",
) -> list[str]:
    """Return the list of grounding problems (empty when the answer is grounded).

    `citations` are `(chunk_label, quote)` pairs, `chunks` maps each provided chunk label to
    its text, and `customer_text` is the question plus history (numbers the customer said
    themselves may be repeated back).
    """
    problems: list[str] = []
    cited_text: list[str] = []
    for label, quote in citations:
        chunk_text = chunks.get(label)
        if chunk_text is None:
            problems.append(f"citation refers to {label}, which was not provided")
            continue
        cited_text.append(chunk_text)
        if not quote_supported(quote, chunk_text):
            problems.append(f"the quote for {label} does not appear in {label}")

    allowed = extract_numbers(" ".join(cited_text)) | extract_numbers(customer_text)
    unsupported = sorted(extract_numbers(answer) - allowed)
    if unsupported:
        problems.append(
            "these numbers in the answer do not appear in the cited chunks: "
            + ", ".join(unsupported)
        )
    return problems
