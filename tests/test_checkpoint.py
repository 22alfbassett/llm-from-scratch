"""Tests for gradient checkpointing — forward output and gradients must
match a non-checkpointed run to within numerical tolerance."""

import numpy as np

from llm.config import ModelConfig
from llm.tensor import Tensor, checkpoint, set_seed
from llm.train import cross_entropy
from llm.transformer import Transformer


def test_checkpoint_forward_matches_plain():
    """checkpoint(f)(x) should produce the same output data as f(x)."""
    set_seed(0)
    w = Tensor.randn(4, 4)

    def f(x):
        return (x.matmul(w)).relu()

    x = Tensor.randn(2, 4)
    y_plain = f(x).data
    y_ckpt = checkpoint(f, x).data
    assert np.allclose(np.asarray(y_plain), np.asarray(y_ckpt), atol=1e-6)


def test_checkpoint_gradients_match_plain():
    """Gradients through a checkpointed op must equal the plain version."""
    set_seed(0)
    w = Tensor.randn(4, 4)

    def f(x):
        return (x.matmul(w)).relu()

    # plain path
    x1 = Tensor.randn(3, 4)
    y1 = f(x1).sum()
    y1.backward()
    g_plain = np.asarray(x1.grad).copy()

    # checkpointed path
    x2 = Tensor(x1.data.copy())
    y2 = checkpoint(f, x2).sum()
    y2.backward()
    g_ckpt = np.asarray(x2.grad).copy()

    assert np.allclose(g_plain, g_ckpt, atol=1e-5)


def test_transformer_grad_checkpoint_matches():
    """Full transformer: loss and parameter gradients should match
    whether grad_checkpoint is on or off."""
    set_seed(42)
    cfg_a = ModelConfig(vocab_size=8, n_embd=16, n_head=2, n_layer=2, block_size=4, dropout=0.0)
    model_a = Transformer(cfg_a)

    set_seed(42)
    cfg_b = ModelConfig(
        vocab_size=8,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=4,
        dropout=0.0,
        grad_checkpoint=True,
    )
    model_b = Transformer(cfg_b)

    x = [[1, 2, 3, 4]]
    y = [[2, 3, 4, 5]]

    logits_a = model_a(x)
    loss_a = cross_entropy(logits_a, y)
    loss_a.backward()

    logits_b = model_b(x)
    loss_b = cross_entropy(logits_b, y)
    loss_b.backward()

    assert np.allclose(np.asarray(loss_a.data), np.asarray(loss_b.data), atol=1e-5)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        ga = np.asarray(p_a.grad)
        gb = np.asarray(p_b.grad)
        assert np.allclose(ga, gb, atol=1e-4), "grad mismatch between plain and checkpointed"
