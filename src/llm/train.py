import gc
import logging
import math
import time
from collections.abc import Iterator
from typing import Optional, Union

from .data import Dataset
from .scheduler import CosineWithWarmup
from .tensor import DTYPE, STABLE_DTYPE, Tensor, no_grad, np
from .transformer import Transformer

logger = logging.getLogger(__name__)


def cross_entropy(logits: Tensor, targets: Union[list, np.ndarray]) -> Tensor:
    """Stable softmax cross-entropy via log-sum-exp.

    Returns a (1, 1)-shaped Tensor wired into logits' autograd graph.
    """
    targets_arr = np.array(targets)
    B, T, V = logits.shape
    logits_flat = logits.data.astype(STABLE_DTYPE).reshape(B * T, V)
    targets_flat = targets_arr.reshape(B * T)

    max_val = np.max(logits_flat, axis=-1, keepdims=True)
    shifted = logits_flat - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp  # log-softmax
    nll = -log_probs[np.arange(B * T), targets_flat]
    loss_val = float(np.mean(nll))
    out = Tensor([[loss_val]], _prev=(logits,), _op="ce")

    def _backward():
        probs = np.exp(log_probs)
        grad_flat = probs.copy()
        grad_flat[np.arange(B * T), targets_flat] -= 1
        grad = grad_flat.reshape(B, T, V) / (B * T)
        logits.grad += (grad * out.grad[0, 0]).astype(DTYPE)

    out._backward = _backward
    return out


def perplexity(
    model: Transformer,
    dataset: Dataset,
    iters: int = 20,
    batch_size: int = 4,
) -> float:
    """Perplexity = exp(mean cross-entropy) over ``iters`` random batches.

    Lower is better. A perfectly random model over V tokens has
    perplexity = V; a perfect model has perplexity = 1.
    """
    return float(math.exp(estimate_loss(model, dataset, iters=iters, batch_size=batch_size)))


def estimate_loss(
    model: Transformer,
    dataset: Dataset,
    iters: int = 20,
    batch_size: int = 4,
) -> float:
    """Compute mean loss over ``iters`` random batches, under no_grad."""
    model.eval()
    total = 0.0
    with no_grad():
        for _ in range(iters):
            x, y = dataset.get_batch(batch_size)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss_val = loss.data[0][0]
            if hasattr(loss_val, "get"):
                loss_val = loss_val.get()
            total += float(loss_val)
    model.train()
    return total / iters


class Adam:
    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data).astype(STABLE_DTYPE) for p in self.params]
        self.v = [np.zeros_like(p.data).astype(STABLE_DTYPE) for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            p_data_stable = p.data.astype(STABLE_DTYPE)
            p_grad_stable = p.grad.astype(STABLE_DTYPE)

            if self.weight_decay != 0:
                p_data_stable -= self.lr * self.weight_decay * p_data_stable

            self.m[i] = b1 * self.m[i] + (1 - b1) * p_grad_stable
            self.v[i] = b2 * self.v[i] + (1 - b2) * (p_grad_stable**2)

            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            p_data_stable -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p_data_stable.astype(DTYPE)

    # --------------------------------------------------------------
    # Persistence: save/load optimizer state to resume training.
    # --------------------------------------------------------------
    def save(self, path: str) -> None:
        import numpy as _np

        arrs = {"t": _np.array(self.t)}
        for i, mi in enumerate(self.m):
            arrs[f"m_{i}"] = _np.asarray(mi.get() if hasattr(mi, "get") else mi)
        for i, vi in enumerate(self.v):
            arrs[f"v_{i}"] = _np.asarray(vi.get() if hasattr(vi, "get") else vi)
        _np.savez(path, **arrs)

    def load(self, path: str) -> None:
        import numpy as _np

        with _np.load(path) as data:
            self.t = int(data["t"])
            n = len(self.params)
            self.m = [np.array(data[f"m_{i}"]).astype(STABLE_DTYPE) for i in range(n)]
            self.v = [np.array(data[f"v_{i}"]).astype(STABLE_DTYPE) for i in range(n)]


def train_loop(
    model: Transformer,
    dataset: Dataset,
    steps: int = 100,
    lr: float = 3e-4,
    batch_size: int = 4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_steps: int = 100,
    grad_accum_steps: int = 1,
    val_dataset: Optional[Dataset] = None,
    eval_interval: int = 0,
    eval_iters: int = 20,
    log_interval: int = 10,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 0,
    resume_optimizer: Optional[str] = None,
    tokenizer=None,
):
    """Training loop with LR schedule, grad clip, grad accumulation,
    validation, periodic checkpointing, and resume support.
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    loss_history: list[float] = []
    val_history: list[tuple] = []
    schedule = CosineWithWarmup(lr, warmup_steps, steps)
    accum_scale = 1.0 / max(1, grad_accum_steps)

    start_step = 0
    if resume_optimizer is not None:
        logger.info("resuming optimizer state from %s", resume_optimizer)
        optimizer.load(resume_optimizer)
        start_step = min(optimizer.t, steps)

    tokens_per_step = batch_size * grad_accum_steps * model.block_size
    t_start = time.time()

    for step in range(start_step, steps):
        curr_lr = schedule(step)
        optimizer.lr = curr_lr

        model.zero_grad()
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = dataset.get_batch(batch_size)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss_val = loss.data[0][0]
            if hasattr(loss_val, "get"):
                loss_val = loss_val.get()
            step_loss += float(loss_val)
            loss.backward()
            if grad_accum_steps > 1:
                for p in model.parameters():
                    p.grad = (p.grad.astype(STABLE_DTYPE) * accum_scale).astype(DTYPE)
        step_loss /= grad_accum_steps
        loss_history.append(step_loss)

        if grad_clip > 0:
            total_norm_sq = 0.0
            for p in model.parameters():
                total_norm_sq += float(np.sum(p.grad.astype(STABLE_DTYPE) ** 2))
            total_norm = math.sqrt(total_norm_sq)
            clip_coef = grad_clip / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in model.parameters():
                    p.grad = (p.grad.astype(STABLE_DTYPE) * clip_coef).astype(DTYPE)

        optimizer.step()
        gc.collect()

        if step % log_interval == 0 or step == steps - 1:
            elapsed = max(1e-6, time.time() - t_start)
            done = step - start_step + 1
            tok_s = done * tokens_per_step / elapsed
            eta = (steps - step - 1) * elapsed / done
            logger.info(
                "step %4d/%d | loss: %.4f | lr: %.6f | %6.0f tok/s | eta %dm %ds",
                step,
                steps,
                step_loss,
                curr_lr,
                tok_s,
                int(eta // 60),
                int(eta % 60),
            )

        if (
            eval_interval > 0
            and val_dataset is not None
            and (step % eval_interval == 0 or step == steps - 1)
        ):
            val_loss = estimate_loss(model, val_dataset, iters=eval_iters, batch_size=batch_size)
            val_history.append((step, val_loss))
            logger.info("  val loss @ step %d: %.4f", step, val_loss)

        if (
            checkpoint_path is not None
            and checkpoint_interval > 0
            and step > 0
            and (step % checkpoint_interval == 0 or step == steps - 1)
        ):
            logger.info("saving checkpoint to %s", checkpoint_path)
            model.save(checkpoint_path, tokenizer=tokenizer)
            optimizer.save(checkpoint_path + ".opt.npz")

    return {"train": loss_history, "val": val_history, "optimizer": optimizer}
