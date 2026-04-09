import math

import numpy as np

from llm.tensor import Tensor


def approx(a, b, tol=1e-5):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.allclose(a, b, atol=tol)


def test_elementwise_ops():
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = Tensor([[0.5, 0.5], [0.5, 0.5]])
    assert approx((t1 + t2).data, [[1.5, 2.5], [3.5, 4.5]])
    assert approx((t1 * t2).data, [[0.5, 1.0], [1.5, 2.0]])
    assert approx((1.0 + t1).data, [[2.0, 3.0], [4.0, 5.0]])
    assert approx((1.0 - t1).data, [[0.0, -1.0], [-2.0, -3.0]])
    assert approx((2.0 * t1).data, [[2.0, 4.0], [6.0, 8.0]])


def test_matmul_and_reshape():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    assert approx(a.matmul(b).data, [[58.0, 64.0], [139.0, 154.0]])
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert approx(t.sum().data, 10.0)
    assert approx(t.reshape(1, 4).data, [[1.0, 2.0, 3.0, 4.0]])


def test_activations():
    assert approx(Tensor([[-1.0, 2.0]]).relu().data, [[0.0, 2.0]])
    assert approx(Tensor([[0.0, math.log(3.0)]]).softmax().data, [[0.25, 0.75]])
    assert approx(Tensor([[0.0]]).sigmoid().data, [[0.5]])


def test_split():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    parts = t.split(2, axis=0)
    assert approx(parts[0].data, [[1.0, 2.0]])
    assert approx(parts[1].data, [[3.0, 4.0]])
