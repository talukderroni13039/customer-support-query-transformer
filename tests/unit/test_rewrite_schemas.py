import pytest
from pydantic import ValidationError

from src.domains.support_search.prompts.rewrite_prompt import (
    PROMPT_VERSION,
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_user_message,
)
from src.domains.support_search.schemas import RewrittenQuery


def test_valid_rewrite_collapses_whitespace() -> None:
    result = RewrittenQuery(
        needs_retrieval=True, standalone_query="  Return window for\n the TB-PRO-22  jacket "
    )
    assert result.standalone_query == "Return window for the TB-PRO-22 jacket"
    assert result.from_cache is False


def test_small_talk_allows_empty_query() -> None:
    result = RewrittenQuery(needs_retrieval=False)
    assert result.standalone_query == ""


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_retrieval_requires_a_query(query: str) -> None:
    with pytest.raises(ValidationError, match="standalone_query must not be empty"):
        RewrittenQuery(needs_retrieval=True, standalone_query=query)


def test_query_length_is_capped() -> None:
    with pytest.raises(ValidationError):
        RewrittenQuery(needs_retrieval=True, standalone_query="x" * 501)


def test_needs_retrieval_is_required() -> None:
    with pytest.raises(ValidationError):
        RewrittenQuery.model_validate({"standalone_query": "q"})


def test_from_cache_is_hidden_from_the_llm_schema_and_from_dumps() -> None:
    schema = RewrittenQuery.model_json_schema()
    assert set(schema["properties"]) == {"needs_retrieval", "standalone_query"}
    assert schema["required"] == ["needs_retrieval"]
    dumped = RewrittenQuery(needs_retrieval=True, standalone_query="q", from_cache=True)
    assert dumped.model_dump() == {"needs_retrieval": True, "standalone_query": "q"}


def test_prompt_states_the_rewrite_rules() -> None:
    prompt = REWRITE_SYSTEM_PROMPT.lower()
    for rule in (
        "pronouns",
        "exactly as written",
        "do not answer",
        "needs_retrieval",
        "correction",
    ):
        assert rule in prompt
    assert PROMPT_VERSION


def test_user_message_numbers_history_oldest_first_without_roles() -> None:
    message = build_rewrite_user_message("How long do I have to return it?", ["first", "second"])
    assert "1. first\n2. second" in message
    assert message.index("1. first") < message.index("How long do I have to return it?")
    assert "role" not in message.lower()
    assert "user:" not in message.lower()
