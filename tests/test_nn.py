import math

import numpy as np

from llm.nn import Embedding, LayerNorm, Linear, RMSNorm, SiLU
from llm.tensor import DTYPE, Tensor
from llm.tensor import np as dnp


def test_linear_forward():
    lin = Linear(2, 3)
    lin.weight.data = dnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    lin.bias.data = dnp.array([[0.1, 0.2, 0.3]], dtype=DTYPE)
    out = lin(Tensor([[1.0, 0.0]]))
    assert np.allclose(out.data, [[1.1, 2.2, 3.3]], atol=1e-5)


def test_embedding():
    emb = Embedding(10, 4)
    emb.weight.data[3] = dnp.array([0.1, 0.2, 0.3, 0.4], dtype=DTYPE)
    assert np.allclose(emb([3]).data, [[0.1, 0.2, 0.3, 0.4]], atol=1e-5)


def test_layernorm_rmsnorm():
    ln = LayerNorm(2)
    ln.gamma.data = dnp.array([[1.0, 1.0]], dtype=DTYPE)
    ln.beta.data = dnp.array([[0.0, 0.0]], dtype=DTYPE)
    assert np.allclose(ln(Tensor([[10.0, 20.0]])).data, [[-1.0, 1.0]], atol=1e-4)

    rmsn = RMSNorm(2)
    rmsn.gamma.data = dnp.array([[1.0, 1.0]], dtype=DTYPE)
    expected = [[3.0 / math.sqrt(12.5), 4.0 / math.sqrt(12.5)]]
    assert np.allclose(rmsn(Tensor([[3.0, 4.0]])).data, expected, atol=1e-4)


def test_silu():
    silu = SiLU()
    assert np.allclose(silu(Tensor([[0.0]])).data, [[0.0]], atol=1e-5)
    assert np.allclose(silu(Tensor([[1.0]])).data, [[1.0 / (1.0 + math.exp(-1.0))]], atol=1e-4)


def test_auto_parameter_registration():
    """Module.parameters() should auto-discover tensors and nested modules."""
    from llm.nn import Module

    class Toy(Module):
        def __init__(self):
            self.lin1 = Linear(2, 3)
            self.lin2 = Linear(3, 4)
            self.extras = [Linear(4, 4), Linear(4, 2)]

    t = Toy()
    params = list(t.parameters())
    # 4 Linears * (weight+bias) = 8
    assert len(params) == 8
