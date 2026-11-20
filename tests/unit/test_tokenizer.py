from src.keyword_search.tokenizer import STOP_WORDS, tokenize


def test_lowercases() -> None:
    assert tokenize("Waterproof JACKET") == ["waterproof", "jacket"]


def test_hyphenated_code_stays_one_token() -> None:
    assert tokenize("TB-PRO-22") == ["tb-pro-22"]
    assert tokenize("Is TB-PRO-22 waterproof?") == ["tb-pro-22", "waterproof"]
    assert tokenize("SS-110 vs SS-111") == ["ss-110", "vs", "ss-111"]


def test_punctuation_stripped() -> None:
    assert tokenize("Refund?! (store-credit), e-mail: now.") == [
        "refund",
        "store",
        "credit",
        "e",
        "mail",
        "now",
    ]
    assert tokenize("...!!!") == []


def test_stop_words_removed() -> None:
    assert tokenize("How long do I have to return it?") == ["long", "return"]
    assert all(word not in STOP_WORDS for word in tokenize("the jacket is on the way to you"))


def test_hyphenated_word_without_digits_matches_unhyphenated_spelling() -> None:
    assert tokenize("water-resistant") == tokenize("water resistant") == ["water", "resist"]


def test_same_output_for_chunks_and_queries() -> None:
    chunk_text = "Trailblazer Pro — Care: The TB-PRO-22 is machine washable."
    query = "trailblazer pro care the tb-pro-22 is machine washable"
    assert tokenize(chunk_text) == tokenize(query)


def test_words_are_stemmed_so_singular_matches_plural_and_tenses() -> None:
    assert tokenize("jackets") == tokenize("jacket") == ["jacket"]
    assert tokenize("returned") == tokenize("returns") == tokenize("return") == ["return"]
    assert tokenize("shipping") == tokenize("ship")
    assert tokenize("How long do I have to return a jacket") == ["long", "return", "jacket"]


def test_anything_with_a_digit_is_never_stemmed() -> None:
    assert tokenize("TB-PRO-22 SS-110 20000mm 60-day size9s") == [
        "tb-pro-22",
        "ss-110",
        "20000mm",
        "60-day",
        "size9s",
    ]
