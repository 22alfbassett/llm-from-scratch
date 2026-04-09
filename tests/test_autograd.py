"""Finite-difference gradient checks for Tensor autograd."""

import numpy as np
import pytest

from llm.tensor import Tensor


def numerical_grad(f, x, eps=1e-3):
    """Return numerical gradient of scalar f(x) wrt each entry of x (numpy array)."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = float(f(x))
        x[idx] = orig - eps
        fm = float(f(x))
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def analytic_grad(build, x_np):
    """Build a tensor from x_np, call build(t), sum, backward, return t.grad."""
    t = Tensor(x_np.copy())
    out = build(t)
    out.sum().backward()
    return np.asarray(t.grad, dtype=np.float64)


@pytest.mark.parametrize(
    "op,builder",
    [
        ("add", lambda t: t + 3.0),
        ("mul", lambda t: t * 2.5),
        ("sub_r", lambda t: 5.0 - t),
        ("pow", lambda t: t**2),
        ("relu", lambda t: t.relu()),
        ("sigmoid", lambda t: t.sigmoid()),
        ("tanh", lambda t: t.tanh()),
        ("exp", lambda t: t.exp()),
        ("sum", lambda t: t.sum()),
        ("reshape", lambda t: t.reshape(4)),
    ],
)
def test_unary_gradcheck(op, builder):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 2)).astype(np.float64)
    if op == "exp":
        x = x * 0.5  # keep magnitudes small
    analytic = analytic_grad(builder, x)
    numeric = numerical_grad(lambda v: float(builder(Tensor(v)).sum().data), x.copy())
    assert np.allclose(analytic, numeric, atol=1e-2), (analytic, numeric)


def test_matmul_gradcheck():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 3)).astype(np.float64)
    B = rng.standard_normal((3, 2)).astype(np.float64)

    def f_a(x):
        return float((Tensor(x).matmul(Tensor(B))).sum().data)

    def f_b(x):
        return float((Tensor(A).matmul(Tensor(x))).sum().data)

    a_t = Tensor(A.copy())
    b_t = Tensor(B.copy())
    (a_t.matmul(b_t)).sum().backward()
    assert np.allclose(a_t.grad, numerical_grad(f_a, A.copy()), atol=1e-2)
    assert np.allclose(b_t.grad, numerical_grad(f_b, B.copy()), atol=1e-2)


def test_softmax_gradcheck():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 4)).astype(np.float64)

    def scalar_fn(v):
        return float((Tensor(v).softmax() * Tensor([[1.0, 2.0, 3.0, 4.0]])).sum().data)

    t = Tensor(x.copy())
    (t.softmax() * Tensor([[1.0, 2.0, 3.0, 4.0]])).sum().backward()
    numeric = numerical_grad(scalar_fn, x.copy())
    assert np.allclose(t.grad, numeric, atol=1e-2)
