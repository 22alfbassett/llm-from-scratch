from llm.data import BPETokenizer, Tokenizer


def test_char_tokenizer_empty_roundtrip():
    tok = Tokenizer("abc")
    assert tok.decode(tok.encode("")) == ""


def test_bpe_unicode_roundtrip():
    tok = BPETokenizer()
    tok.train("héllo wörld 🌍 héllo wörld 🌍", vocab_size=280)
    text = "héllo 🌍"
    assert tok.decode(tok.encode(text)) == text


def test_char_tokenizer_unknown_char():
    tok = Tokenizer("abc")
    encoded = tok.encode("abxc")
    assert len(encoded) == 4
    # 'x' is unknown, should map to the unk token and decode to the replacement char
    decoded = tok.decode(encoded)
    assert decoded[0] == "a"
    assert decoded[1] == "b"
    assert decoded[2] == Tokenizer.UNK
    assert decoded[3] == "c"


def test_bpe_empty_encode():
    tok = BPETokenizer()
    tok.train("abcabc", vocab_size=260)
    assert tok.encode("") == []
    assert tok.decode([]) == ""
