import os

import numpy as np
import pytest

from llm.config import ModelConfig
from llm.transformer import Transformer


def _small_cfg(vocab=10, tie=True, act="swiglu", norm="rmsnorm"):
    return ModelConfig(
        vocab_size=vocab,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=8,
        dropout=0.0,
        tie_weights=tie,
        act=act,
        norm=norm,
    )


def test_forward_shape():
    model = Transformer(_small_cfg())
    out = model([[1, 2, 3]])
    assert out.shape == (1, 3, 10)


def test_generate_extends_idx():
    model = Transformer(_small_cfg())
    idx = [1, 2]
    list(model.generate(idx, max_new_tokens=3))
    assert len(idx) == 5


@pytest.mark.parametrize("act", ["swiglu", "gelu", "relu"])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("tie", [True, False])
def test_variants(act, norm, tie):
    model = Transformer(_small_cfg(tie=tie, act=act, norm=norm))
    out = model([[1, 2, 3, 4]])
    assert out.shape == (1, 4, 10)


def test_checkpoint_roundtrip(tmp_path):
    model = Transformer(_small_cfg())
    path = str(tmp_path / "ckpt.npz")
    model.save(path)
    assert os.path.exists(path)
    # from_checkpoint should reconstruct architecture from embedded config
    model2, tok = Transformer.from_checkpoint(path)
    assert tok is None  # no tokenizer was saved
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        a = np.asarray(p1.data)
        b = np.asarray(p2.data)
        assert np.allclose(a, b, atol=1e-6)
    # and forward outputs should match
    x = [[1, 2, 3]]
    y1 = np.asarray(model(x).data)
    y2 = np.asarray(model2(x).data)
    assert np.allclose(y1, y2, atol=1e-5)


def test_checkpoint_with_tokenizer(tmp_path):
    from llm.data import BPETokenizer

    model = Transformer(_small_cfg())
    tok = BPETokenizer()
    tok.train("hello world hello world", vocab_size=270)

    path = str(tmp_path / "ckpt_tok.npz")
    model.save(path, tokenizer=tok)

    model2, tok2 = Transformer.from_checkpoint(path)
    assert tok2 is not None
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode("hello") == tok.encode("hello")
    assert tok2.decode(tok2.encode("hello world")) == "hello world"


def test_tied_weights_param_count():
    """Tied model should have fewer params than untied (no separate lm_head.weight)."""
    tied = Transformer(_small_cfg(tie=True))
    untied = Transformer(_small_cfg(tie=False))
    n_tied = sum(p.data.size for p in tied.parameters())
    n_untied = sum(p.data.size for p in untied.parameters())
    assert n_untied > n_tied
