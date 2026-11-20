"""Keep untrusted text from imitating the structure of a prompt."""


def escape_prompt_text(text: str) -> str:
    """Neutralise angle brackets in customer or article text before it goes into a prompt.

    The prompts wrap each input in tags such as `<history>` and `<chunks>`. Without escaping,
    a message containing `</history>` or a fake `<chunks>` block could close a section early
    or pose as trusted content. Escaped text still reads the same to the model.
    """
    return text.replace("<", "&lt;").replace(">", "&gt;")
