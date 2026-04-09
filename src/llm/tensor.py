import os
import random as _random
from contextlib import contextmanager
from typing import Union

import numpy as _np

# check for device preference before importing cupy
_device_pref = os.environ.get("LLM_DEVICE", "auto").lower()

USING_GPU = False
if _device_pref != "cpu":
    try:
        import cupy as np

        try:
            # check if a gpu is actually available
            np.cuda.Device(0).use()
            USING_GPU = True
        except Exception:
            np = _np
            USING_GPU = False
    except ImportError:
        np = _np
        USING_GPU = False
else:
    np = _np
    USING_GPU = False

# set default dtype from env or float32
_dtype_str = os.environ.get("LLM_DTYPE", "float32").lower()
if _dtype_str == "float32":
    DTYPE = np.float32
elif _dtype_str == "float64":
    DTYPE = np.float64
elif _dtype_str == "float16":
    DTYPE = np.float16
else:
    DTYPE = np.float32

# use at least float32 for stable operations
STABLE_DTYPE = _np.promote_types(DTYPE, _np.float32)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if active) CuPy RNGs for reproducibility."""
    _random.seed(seed)
    _np.random.seed(seed)
    if USING_GPU:
        np.random.seed(seed)


@contextmanager
def no_grad():
    """Context manager that disables autograd graph construction."""
    prev = Tensor.requires_grad
    Tensor.requires_grad = False
    try:
        yield
    finally:
        Tensor.requires_grad = prev


def checkpoint(fn, *inputs: "Tensor") -> "Tensor":
    """Gradient checkpointing.

    Run ``fn(*inputs)`` once under no_grad so no interior graph is kept
    in memory. During backward, re-run ``fn`` with autograd enabled on
    fresh input copies, call backward on that local graph with the
    saved upstream gradient, and add the resulting gradients back into
    the original inputs.

    Trades compute for memory: the activations inside ``fn`` are never
    stored, only recomputed when needed.
    """
    with no_grad():
        out_nograd = fn(*inputs)

    out = Tensor(
        out_nograd.data,
        _prev=tuple(inputs) if Tensor.requires_grad else (),
        _op="checkpoint",
    )

    if Tensor.requires_grad:

        def _backward():
            # Recompute the segment with autograd enabled, using copies
            # of the inputs so we don't pollute any existing graph.
            input_copies = [Tensor(inp.data) for inp in inputs]
            y = fn(*input_copies)
            y.backward(grad=out.grad)
            for orig, copy in zip(inputs, input_copies):
                orig.grad += copy.grad

        out._backward = _backward

    return out


class Tensor:
    __slots__ = ("data", "shape", "grad", "_prev", "_op", "_backward")
    requires_grad = True

    def __init__(
        self,
        data: Union[list, np.ndarray, float, int],
        _prev: tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(DTYPE)
        elif isinstance(data, (float, int)):
            self.data = np.array([data], dtype=DTYPE).reshape(())
        else:
            # move to current device
            self.data = np.array(data, dtype=DTYPE)

        self.shape: tuple[int, ...] = self.data.shape
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._prev = set(_prev) if Tensor.requires_grad else set()
        self._op = _op
        self._backward = lambda: None

    @staticmethod
    def zeros(*shape: int) -> "Tensor":
        return Tensor(np.zeros(shape))

    @staticmethod
    def randn(*shape: int) -> "Tensor":
        return Tensor(np.random.randn(*shape))

    @staticmethod
    def cat(tensors: list["Tensor"], axis: int = 1) -> "Tensor":
        new_data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(new_data, _prev=tuple(tensors) if Tensor.requires_grad else (), _op="cat")

        if Tensor.requires_grad:

            def _backward():
                offset = 0
                for t in tensors:
                    size = t.shape[axis]
                    slc = [slice(None)] * out.grad.ndim
                    slc[axis] = slice(offset, offset + size)
                    t.grad += out.grad[tuple(slc)]
                    offset += size

            out._backward = _backward
        return out

    def _reduce_grad(self, grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        if grad.shape == target_shape:
            return grad

        # aggregate grads over broadcast dims
        n_dims_grad = grad.ndim
        n_dims_target = len(target_shape)
        sum_axes = list(range(n_dims_grad - n_dims_target))
        offset = n_dims_grad - n_dims_target
        for i, dim in enumerate(target_shape):
            if dim == 1:
                sum_axes.append(i + offset)

        if sum_axes:
            grad = grad.sum(axis=tuple(sum_axes), keepdims=True)
            if n_dims_grad > n_dims_target:
                grad = grad.reshape(target_shape)
        return grad

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, (float, int)):
            out = Tensor(
                self.data + other,
                _prev=(self,) if Tensor.requires_grad else (),
                _op="+",
            )
            if Tensor.requires_grad:

                def _backward():
                    self.grad += self._reduce_grad(out.grad, self.data.shape)

                out._backward = _backward
            return out

        out = Tensor(
            self.data + other.data,
            _prev=(self, other) if Tensor.requires_grad else (),
            _op="+",
        )
        if Tensor.requires_grad:

            def _backward():
                self.grad += self._reduce_grad(out.grad, self.data.shape)
                other.grad += self._reduce_grad(out.grad, other.data.shape)

            out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self + other

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self + (other * -1.0 if isinstance(other, Tensor) else -other)

    def __rsub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return (self * -1.0) + other

    def __pow__(self, other: Union[float, int]) -> "Tensor":
        # stable calculation
        data_stable = self.data.astype(STABLE_DTYPE)
        out_data = (data_stable**other).astype(DTYPE)
        out = Tensor(out_data, _prev=(self,) if Tensor.requires_grad else (), _op=f"**{other}")
        if Tensor.requires_grad:

            def _backward():
                self_data_stable = self.data.astype(STABLE_DTYPE)
                out_grad_stable = out.grad.astype(STABLE_DTYPE)
                grad_stable = (other * (self_data_stable ** (other - 1.0))) * out_grad_stable
                self.grad += self._reduce_grad(grad_stable.astype(DTYPE), self.data.shape)

            out._backward = _backward
        return out

    def mean(self, axis: int = -1) -> "Tensor":
        # stable calculation
        data_stable = self.data.astype(STABLE_DTYPE)
        out_data = data_stable.mean(axis=axis, keepdims=True).astype(DTYPE)
        out = Tensor(out_data, _prev=(self,) if Tensor.requires_grad else (), _op="mean")
        if Tensor.requires_grad:

            def _backward():
                n = self.data.shape[axis]
                grad_stable = out.grad.astype(STABLE_DTYPE)
                self.grad += ((1.0 / n) * grad_stable).astype(DTYPE)

            out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, (float, int)):
            out = Tensor(
                self.data * other,
                _prev=(self,) if Tensor.requires_grad else (),
                _op="*",
            )
            if Tensor.requires_grad:

                def _backward():
                    self.grad += self._reduce_grad(
                        (other * out.grad.astype(STABLE_DTYPE)).astype(DTYPE),
                        self.data.shape,
                    )

                out._backward = _backward
            return out

        out = Tensor(
            self.data * other.data,
            _prev=(self, other) if Tensor.requires_grad else (),
            _op="*",
        )
        if Tensor.requires_grad:

            def _backward():
                # stable multiplication
                self_data_stable = self.data.astype(STABLE_DTYPE)
                other_data_stable = other.data.astype(STABLE_DTYPE)
                out_grad_stable = out.grad.astype(STABLE_DTYPE)
                self.grad += self._reduce_grad(
                    (other_data_stable * out_grad_stable).astype(DTYPE), self.data.shape
                )
                other.grad += self._reduce_grad(
                    (self_data_stable * out_grad_stable).astype(DTYPE), other.data.shape
                )

            out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self * other

    def matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data @ other.data,
            _prev=(self, other) if Tensor.requires_grad else (),
            _op="@",
        )
        if Tensor.requires_grad:

            def _backward():
                # stable matmul
                self_data_stable = self.data.astype(STABLE_DTYPE)
                other_data_stable = other.data.astype(STABLE_DTYPE)
                out_grad_stable = out.grad.astype(STABLE_DTYPE)
                self.grad += self._reduce_grad(
                    (out_grad_stable @ np.swapaxes(other_data_stable, -1, -2)).astype(DTYPE),
                    self.data.shape,
                )
                other.grad += self._reduce_grad(
                    (np.swapaxes(self_data_stable, -1, -2) @ out_grad_stable).astype(DTYPE),
                    other.data.shape,
                )

            out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(shape),
            _prev=(self,) if Tensor.requires_grad else (),
            _op="reshape",
        )
        if Tensor.requires_grad:

            def _backward():
                self.grad += out.grad.reshape(self.data.shape)

            out._backward = _backward
        return out

    def transpose(self, axis1: int = -1, axis2: int = -2) -> "Tensor":
        out = Tensor(
            np.swapaxes(self.data, axis1, axis2),
            _prev=(self,) if Tensor.requires_grad else (),
            _op="T",
        )
        if Tensor.requires_grad:

            def _backward():
                self.grad += np.swapaxes(out.grad, axis1, axis2)

            out._backward = _backward
        return out

    def permute(self, *axes: int) -> "Tensor":
        out = Tensor(
            np.transpose(self.data, axes),
            _prev=(self,) if Tensor.requires_grad else (),
            _op="permute",
        )
        if Tensor.requires_grad:

            def _backward():
                inv_axes = np.argsort(np.array(axes))
                self.grad += np.transpose(out.grad, tuple(inv_axes.tolist()))

            out._backward = _backward
        return out

    def split(self, n: int, axis: int = 0) -> list["Tensor"]:
        size = self.data.shape[axis] // n
        parts = np.split(self.data, n, axis=axis)
        outputs = []
        for i, p in enumerate(parts):
            out = Tensor(p, _prev=(self,) if Tensor.requires_grad else (), _op=f"split_{i}")
            if Tensor.requires_grad:

                def _backward(i=i, out=out):
                    slc = [slice(None)] * self.data.ndim
                    slc[axis] = slice(i * size, (i + 1) * size)
                    self.grad[tuple(slc)] += out.grad

                out._backward = _backward
            outputs.append(out)
        return outputs

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0, self.data),
            _prev=(self,) if Tensor.requires_grad else (),
            _op="relu",
        )
        if Tensor.requires_grad:

            def _backward():
                self.grad += (self.data > 0) * out.grad

            out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        # stable calculation
        data_stable = self.data.astype(STABLE_DTYPE)
        out_data = np.exp(data_stable).astype(DTYPE)
        out = Tensor(out_data, _prev=(self,) if Tensor.requires_grad else (), _op="exp")
        if Tensor.requires_grad:

            def _backward():
                out_data_stable = out.data.astype(STABLE_DTYPE)
                out_grad_stable = out.grad.astype(STABLE_DTYPE)
                self.grad += (out_data_stable * out_grad_stable).astype(DTYPE)

            out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        # stable calculation
        data_stable = self.data.astype(STABLE_DTYPE)
        out_data = np.tanh(data_stable).astype(DTYPE)
        out = Tensor(out_data, _prev=(self,) if Tensor.requires_grad else (), _op="tanh")
        if Tensor.requires_grad:

            def _backward():
                out_data_stable = out.data.astype(STABLE_DTYPE)
                grad_stable = out.grad.astype(STABLE_DTYPE)
                self.grad += ((1.0 - out_data_stable**2) * grad_stable).astype(DTYPE)

            out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # stable calculation
        data_stable = self.data.astype(STABLE_DTYPE)
        out_data = (1.0 / (1.0 + np.exp(-data_stable))).astype(DTYPE)
        out = Tensor(out_data, _prev=(self,) if Tensor.requires_grad else (), _op="sigmoid")
        if Tensor.requires_grad:

            def _backward():
                out_data_stable = out.data.astype(STABLE_DTYPE)
                grad_stable = out.grad.astype(STABLE_DTYPE)
                self.grad += (out_data_stable * (1.0 - out_data_stable) * grad_stable).astype(DTYPE)

            out._backward = _backward
        return out

    def sum(self, axis: Union[int, tuple[int, ...]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            _prev=(self,) if Tensor.requires_grad else (),
            _op="sum",
        )
        if Tensor.requires_grad:

            def _backward():
                self.grad += np.broadcast_to(out.grad, self.data.shape)

            out._backward = _backward
        return out

    def softmax(self) -> "Tensor":
        # stable calculation
        data_stable = self.data.astype(STABLE_DTYPE)
        max_val = np.max(data_stable, axis=-1, keepdims=True)
        exps = np.exp(data_stable - max_val)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Tensor(
            probs.astype(DTYPE),
            _prev=(self,) if Tensor.requires_grad else (),
            _op="softmax",
        )
        if Tensor.requires_grad:

            def _backward():
                probs_stable = out.data.astype(STABLE_DTYPE)
                grad_stable = out.grad.astype(STABLE_DTYPE)
                row_sum = np.sum(probs_stable * grad_stable, axis=-1, keepdims=True)
                self.grad += (probs_stable * (grad_stable - row_sum)).astype(DTYPE)

            out._backward = _backward
        return out

    def backward(self, grad: "np.ndarray | None" = None) -> None:
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if v in visited:
                continue
            if processed:
                visited.add(v)
                topo.append(v)
            else:
                stack.append((v, True))
                for child in v._prev:
                    if child not in visited:
                        stack.append((child, False))

        for v in topo:
            v.grad = np.zeros_like(v.data)

        if grad is not None:
            self.grad = np.asarray(grad).astype(DTYPE).reshape(self.data.shape)
        elif self.shape in [(1, 1), (1,), ()]:
            self.grad.fill(1.0)
        else:
            self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        # move to cpu for printing
        data_cpu = self.data.get() if hasattr(self.data, "get") else self.data
        return f"Tensor({data_cpu})"
