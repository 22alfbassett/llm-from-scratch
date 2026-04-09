from llm.data import BPETokenizer, Dataset, Tokenizer, tokenizer_from_dict


def test_char_tokenizer_roundtrip():
    tok = Tokenizer("abc")
    assert tok.decode(tok.encode("aba")) == "aba"


def test_char_tokenizer_dict_roundtrip():
    tok = Tokenizer("hello world")
    d = tok.to_dict()
    tok2 = Tokenizer.from_dict(d)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode("hello") == tok.encode("hello")
    assert tok2.decode(tok2.encode("hello")) == "hello"


def test_bpe_tokenizer_roundtrip():
    bpe = BPETokenizer()
    bpe.train("ababacacab", vocab_size=259)
    assert bpe.decode(bpe.encode("ababacacab")) == "ababacacab"


def test_bpe_dict_roundtrip():
    bpe = BPETokenizer()
    bpe.train("hello world hello world", vocab_size=270)
    d = bpe.to_dict()
    bpe2 = BPETokenizer.from_dict(d)
    assert bpe2.vocab_size == bpe.vocab_size
    assert bpe2.encode("hello") == bpe.encode("hello")
    assert bpe2.decode(bpe2.encode("hello world")) == "hello world"


def test_tokenizer_from_dict_char():
    tok = Tokenizer("abc")
    d = tok.to_dict()
    tok2 = tokenizer_from_dict(d)
    assert isinstance(tok2, Tokenizer)
    assert tok2.encode("abc") == tok.encode("abc")


def test_tokenizer_from_dict_bpe():
    bpe = BPETokenizer()
    bpe.train("hello world hello world", vocab_size=270)
    d = bpe.to_dict()
    tok2 = tokenizer_from_dict(d)
    assert isinstance(tok2, BPETokenizer)
    assert tok2.encode("hello") == bpe.encode("hello")


def test_bpe_save_load(tmp_path):
    bpe = BPETokenizer()
    bpe.train("hello world hello world", vocab_size=270)
    path = str(tmp_path / "tok.json")
    bpe.save(path)

    bpe2 = BPETokenizer()
    bpe2.load(path)
    assert bpe2.encode("hello") == bpe.encode("hello")


def test_dataset_batch_shapes():
    ds = Dataset(list(range(200)), block_size=16)
    x, y = ds.get_batch(batch_size=4)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y should be x shifted by one at every position (since data is 0..199)
    import numpy as np

    assert np.all(np.asarray(y) == np.asarray(x) + 1)
