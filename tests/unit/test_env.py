import pytest

from src.utils.env import env_bool, env_float, env_int, env_str


def test_defaults_when_missing_or_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_VALUE", raising=False)
    assert env_str("X_VALUE", "d") == "d"
    assert env_int("X_VALUE", 3) == 3
    assert env_float("X_VALUE", 1.5) == 1.5
    assert env_bool("X_VALUE", True) is True
    monkeypatch.setenv("X_VALUE", "  ")
    assert env_str("X_VALUE", "d") == "d"
    assert env_int("X_VALUE", 3) == 3
    assert env_float("X_VALUE", 1.5) == 1.5
    assert env_bool("X_VALUE", False) is False


def test_parses_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_VALUE", " 42 ")
    assert env_str("X_VALUE", "d") == "42"
    assert env_int("X_VALUE", 3) == 42
    assert env_float("X_VALUE", 1.5) == 42.0
    monkeypatch.setenv("X_VALUE", "TRUE")
    assert env_bool("X_VALUE", False) is True
    monkeypatch.setenv("X_VALUE", "off")
    assert env_bool("X_VALUE", True) is False


@pytest.mark.parametrize("reader", [env_int, env_float, env_bool])
def test_invalid_values_raise(monkeypatch: pytest.MonkeyPatch, reader) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("X_VALUE", "not-valid")
    with pytest.raises(ValueError, match="X_VALUE"):
        reader("X_VALUE", 1)
