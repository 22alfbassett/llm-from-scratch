import numpy as np
import pytest

from llm.attention import MultiHeadAttention
from llm.config import ModelConfig
from llm.tensor import Tensor
from llm.transformer import Transformer


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_gqa_output_shape(n_kv_head):
    mha = MultiHeadAttention(n_embd=16, n_head=4, n_kv_head=n_kv_head)
    x = Tensor.randn(2, 5, 16)
    y = mha(x)
    assert y.shape == (2, 5, 16)


def test_gqa_has_fewer_kv_params_than_mha():
    mha = MultiHeadAttention(n_embd=32, n_head=8, n_kv_head=8)
    gqa = MultiHeadAttention(n_embd=32, n_head=8, n_kv_head=2)
    n_mha = sum(p.data.size for p in mha.parameters())
    n_gqa = sum(p.data.size for p in gqa.parameters())
    assert n_gqa < n_mha


def test_gqa_transformer_end_to_end():
    cfg = ModelConfig(
        vocab_size=10,
        n_embd=16,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=8,
        dropout=0.0,
    )
    model = Transformer(cfg)
    out = model([[1, 2, 3]])
    assert out.shape == (1, 3, 10)


def test_gqa_invalid_divisor_raises():
    with pytest.raises(AssertionError):
        MultiHeadAttention(n_embd=16, n_head=4, n_kv_head=3)


def test_mqa_extreme():
    """n_kv_head=1 is multi-query attention (all heads share one K/V head)."""
    mha = MultiHeadAttention(n_embd=16, n_head=4, n_kv_head=1)
    y = mha(Tensor.randn(1, 3, 16))
    assert y.shape == (1, 3, 16)
    # Verify it's causal: perturbing last token must not affect earlier ones.
    x1 = Tensor.randn(1, 4, 16)
    x2 = Tensor(x1.data.copy())
    x2.data[0, 3, :] += 5.0
    y1 = np.asarray(mha(x1).data)
    y2 = np.asarray(mha(x2).data)
    assert np.allclose(y1[0, :3], y2[0, :3], atol=1e-4)
