"""Grounding check for generated answers. Pure functions: no I/O, no model calls.

An answer is grounded when every citation points at a chunk that was really provided, every
quote can be found in that chunk, and every number and product code in the answer has a
legitimate source. The check is deliberately lenient about wording: it is there to catch
invented facts, not harmless differences in punctuation or spacing.

Where a number may come from:
- a plain number (a price, a number of days): the cited chunks, or the customer's latest
  message ("my order of 25 jackets"). NOT the earlier conversation: a message there claiming
  "refunds are free for 90 days" must not make "90 days" acceptable in an answer.
- a product or order code (TB-PRO-22, BC-400): the cited chunks or anywhere in the
  conversation, because naming the product the customer asked about is not a new fact.
- a number that is part of a name ("Ridgeline 65", "Basecamp 4"): the same name appearing in
  the conversation, or the number appearing in the title of any provided chunk.
"""

import re

QUOTE_MIN_TOKEN_COVERAGE = 0.8
QUOTE_MIN_TOKENS_FOR_FUZZY = 4

_NON_WORD = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Letters, then hyphen-joined parts, ending in a part that contains a digit: TB-PRO-22, KB-012.
_CODE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*-[A-Za-z]*\d[A-Za-z0-9]*\b")
# A capitalised word followed by a number, as in a product name: Ridgeline 65.
_NAMED_NUMBER = re.compile(r"\b([A-Z][A-Za-z]+)\s+(\d+(?:[.,]\d+)*)\b")


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


def extract_codes(text: str) -> set[str]:
    """Product / order codes such as `TB-PRO-22`, lowercased."""
    return {match.lower() for match in _CODE.findall(text)}


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
    question_text: str = "",
    history_text: str = "",
    titles_text: str = "",
) -> list[str]:
    """Return the list of grounding problems (empty when the answer is grounded).

    `citations` are `(chunk_label, quote)` pairs and `chunks` maps each provided chunk label
    to its text. `question_text` is the customer's latest message and `history_text` the
    earlier conversation (see the module docstring for what each may contribute).
    `titles_text` is the titles of all provided chunks, cited or not.
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

    cited = " ".join(cited_text)
    allowed_codes = (
        extract_codes(cited) | extract_codes(question_text) | extract_codes(history_text)
    )
    unknown_codes = sorted(extract_codes(answer) - allowed_codes)
    if unknown_codes:
        problems.append(
            "these product codes in the answer do not appear in the cited chunks or the "
            "conversation: " + ", ".join(unknown_codes)
        )

    # Codes are judged above, so their digits are not judged again as plain numbers.
    remaining = _CODE.sub(" ", answer)

    # "Ridgeline 65" is a name, not a claim, when the conversation uses the same name.
    conversation = f"{question_text} {history_text}".lower()

    def drop_known_name(match: re.Match[str]) -> str:
        return " " if match.group(0).lower() in conversation else match.group(0)

    remaining = _NAMED_NUMBER.sub(drop_known_name, remaining)
    allowed_numbers = (
        extract_numbers(cited) | extract_numbers(question_text) | extract_numbers(titles_text)
    )
    unsupported = sorted(extract_numbers(remaining) - allowed_numbers)
    if unsupported:
        problems.append(
            "these numbers in the answer do not appear in the cited chunks: "
            + ", ".join(unsupported)
            + " (cite the chunk each one comes from, or leave the unsupported statement out)"
        )
    return problems
