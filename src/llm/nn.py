import math
from collections.abc import Iterator
from typing import Union

from .tensor import DTYPE, STABLE_DTYPE, Tensor, np


class Module:
    training = True

    def train(self):
        self.training = True
        for m in self.submodules():
            m.train()

    def eval(self):
        self.training = False
        for m in self.submodules():
            m.eval()

    def submodules(self) -> Iterator["Module"]:
        for v in self.__dict__.values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        yield item

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0.0)

    def parameters(self) -> Iterator[Tensor]:
        """Auto-discover Tensor and Module attributes, including inside lists."""
        seen: set = set()

        def _visit(val):
            if isinstance(val, Tensor):
                if id(val) not in seen:
                    seen.add(id(val))
                    yield val
            elif isinstance(val, Module):
                for p in val.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield p
            elif isinstance(val, (list, tuple)):
                for item in val:
                    yield from _visit(item)

        for v in self.__dict__.values():
            yield from _visit(v)


class Linear(Module):
    def __init__(self, nin: int, nout: int):
        # xavier init
        std = (1.0 / nin) ** 0.5
        self.weight = Tensor.randn(nin, nout) * std
        self.bias = Tensor.zeros(1, nout)

    def __call__(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight) + self.bias


class Embedding(Module):
    def __init__(self, vocab_size: int, n_embd: int):
        self.weight = Tensor.randn(vocab_size, n_embd) * 0.02

    def __call__(self, idx: Union[list, np.ndarray]) -> Tensor:
        idx_arr = np.array(idx)
        out = Tensor(self.weight.data[idx_arr], _prev=(self.weight,), _op="emb")

        def _backward():
            np.add.at(self.weight.grad, idx_arr, out.grad)

        out._backward = _backward
        return out


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, dim)))
        self.beta = Tensor(np.zeros((1, dim)))

    def __call__(self, x: Tensor) -> Tensor:
        # stable calculation
        x_stable = x.data.astype(STABLE_DTYPE)
        mean = x_stable.mean(axis=-1, keepdims=True)
        var = ((x_stable - mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat_data = (x_stable - mean) * ((var + self.eps) ** -0.5)
        out = Tensor(x_hat_data.astype(DTYPE), _prev=(x,), _op="layernorm")

        def _backward():
            x_stable = x.data.astype(STABLE_DTYPE)
            grad_y_stable = out.grad.astype(STABLE_DTYPE)
            mean = x_stable.mean(axis=-1, keepdims=True)
            var = ((x_stable - mean) ** 2).mean(axis=-1, keepdims=True)
            std_inv = (var + self.eps) ** -0.5
            N = x_stable.shape[-1]
            x_hat = (x_stable - mean) * std_inv
            grad_x = (
                (1.0 / N)
                * std_inv
                * (
                    N * grad_y_stable
                    - np.sum(grad_y_stable, axis=-1, keepdims=True)
                    - x_hat * np.sum(grad_y_stable * x_hat, axis=-1, keepdims=True)
                )
            )
            x.grad += grad_x.astype(DTYPE)

        out._backward = _backward
        return out * self.gamma + self.beta


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, dim)))

    def __call__(self, x: Tensor) -> Tensor:
        # stable calculation
        x_stable = x.data.astype(STABLE_DTYPE)
        rms = (np.mean(x_stable**2, axis=-1, keepdims=True) + self.eps) ** 0.5
        x_hat_data = (x_stable / rms).astype(DTYPE)
        out = Tensor(x_hat_data, _prev=(x,), _op="rmsnorm")

        def _backward():
            x_stable = x.data.astype(STABLE_DTYPE)
            grad_y_stable = out.grad.astype(STABLE_DTYPE)
            rms = (np.mean(x_stable**2, axis=-1, keepdims=True) + self.eps) ** 0.5
            y_stable = x_stable / rms
            grad_x = (1.0 / rms) * (
                grad_y_stable - y_stable * np.mean(y_stable * grad_y_stable, axis=-1, keepdims=True)
            )
            x.grad += grad_x.astype(DTYPE)

        out._backward = _backward
        return out * self.gamma


class SiLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class ReLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        # tanh approximation
        sqrt_2_over_pi = (2.0 / math.pi) ** 0.5
        return 0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x**3)).tanh())


class Dropout(Module):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(DTYPE)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * mask * scale, _prev=(x,), _op="dropout")

        def _backward():
            x.grad += mask * scale * out.grad

        out._backward = _backward
        return out


class Sequential(Module):
    def __init__(self, layers: list[Module]):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
