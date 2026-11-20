"""Output guardrail: mask personal data and links that did not come from the help articles.

Pure functions. This is pattern matching, not a content classifier: it catches contact
details, card-like numbers, and links. It does not judge tone or harmful content; the
grounding check is what keeps the answer to what the articles say.
"""

import re

MASK = "[removed]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w])")


def apply_output_guardrail(answer: str, allowed_text: str) -> tuple[str, list[str]]:
    """Return `(safe_answer, actions)`.

    `allowed_text` is the text of the chunks the answer was built from: an email address,
    link, or phone number that appears there is part of the help content and is kept.
    Card-like numbers are always masked.
    """
    actions: list[str] = []
    allowed = allowed_text.lower()

    def mask_unless_allowed(pattern: re.Pattern[str], action: str, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            found = match.group(0)
            if found.lower() in allowed:
                return found
            if action not in actions:
                actions.append(action)
            return MASK

        return pattern.sub(replace, text)

    def mask_card(match: re.Match[str]) -> str:
        if "masked_card_number" not in actions:
            actions.append("masked_card_number")
        return MASK

    safe = _CARD.sub(mask_card, answer)
    safe = mask_unless_allowed(_EMAIL, "masked_email", safe)
    safe = mask_unless_allowed(_URL, "masked_link", safe)

    def replace_phone(match: re.Match[str]) -> str:
        found = match.group(0)
        digits = sum(ch.isdigit() for ch in found)
        if digits < 9 or found.lower() in allowed:
            return found
        if "masked_phone_number" not in actions:
            actions.append("masked_phone_number")
        return MASK

    safe = _PHONE.sub(replace_phone, safe)
    return safe, actions
