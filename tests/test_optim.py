import numpy as np

from llm.tensor import Tensor
from llm.train import Adam


def test_adam_minimizes_quadratic():
    """Adam should drive a simple (x - 3)^2 bowl toward x = 3."""
    x = Tensor(np.array([[0.0]]))
    opt = Adam([x], lr=0.1)
    for _ in range(200):
        x.grad = 2.0 * (x.data - 3.0)
        opt.step()
    assert abs(float(x.data[0, 0]) - 3.0) < 1e-2
