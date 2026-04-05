from src.tokenizer import TiktokenTokenizer


def test_tiktoken_tokenizer_round_trip_preserves_text():
    tokenizer = TiktokenTokenizer()
    text = "Ciao! Questo è un test con accenti: perché, città, università."

    tokens = tokenizer.encode(text)

    assert isinstance(tokens, list)
    assert tokens
    assert tokenizer.decode(tokens) == text


def test_tiktoken_tokenizer_exposes_end_token():
    tokenizer = TiktokenTokenizer()

    end_token = tokenizer.get_end_token()

    assert isinstance(end_token, int)
    assert end_token >= 0
    assert tokenizer.decode([end_token]) != ""
