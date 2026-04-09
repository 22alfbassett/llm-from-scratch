from typing import Optional

from .nn import Dropout, Linear, Module
from .tensor import DTYPE, Tensor, np


def precompute_rope_freqs(hs: int, max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    # rope freqs
    theta = 1.0 / (10000.0 ** (np.arange(0, hs, 2).astype(DTYPE) / hs))
    m = np.arange(max_seq_len).astype(DTYPE)
    freqs = np.outer(m, theta)
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray, seq_offset: int = 0) -> Tensor:
    B, nh, T, hs = x.shape
    assert hs % 2 == 0

    cos_b = cos[seq_offset : seq_offset + T, :][np.newaxis, np.newaxis, :, :]
    sin_b = sin[seq_offset : seq_offset + T, :][np.newaxis, np.newaxis, :, :]

    x_reshaped = x.data.reshape(B, nh, T, hs // 2, 2)
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]

    out_data = np.zeros_like(x_reshaped)
    out_data[..., 0] = x1 * cos_b - x2 * sin_b
    out_data[..., 1] = x1 * sin_b + x2 * cos_b
    out_data = out_data.reshape(B, nh, T, hs)

    out = Tensor(out_data, _prev=(x,), _op="rope")

    def _backward():
        grad_reshaped = out.grad.reshape(B, nh, T, hs // 2, 2)
        g1, g2 = grad_reshaped[..., 0], grad_reshaped[..., 1]
        dg1 = g1 * cos_b + g2 * sin_b
        dg2 = -g1 * sin_b + g2 * cos_b
        x.grad += np.stack([dg1, dg2], axis=-1).reshape(B, nh, T, hs)

    out._backward = _backward
    return out


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat K or V along the head axis to match the number of query heads.

    Shape: (B, n_kv_head, T, hs) -> (B, n_kv_head * n_rep, T, hs).
    Backward sums the repeated gradients back to the original heads.
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hs = x.shape
    data = np.repeat(x.data, n_rep, axis=1)
    out = Tensor(data, _prev=(x,) if Tensor.requires_grad else (), _op="repeat_kv")

    if Tensor.requires_grad:

        def _backward():
            grad_reshaped = out.grad.reshape(B, n_kv, n_rep, T, hs)
            x.grad += grad_reshaped.sum(axis=2)

        out._backward = _backward
    return out


class MultiHeadAttention(Module):
    """Multi-head self-attention with optional Grouped-Query Attention.

    If ``n_kv_head`` is None or equal to ``n_head``, this is standard MHA.
    If ``n_kv_head < n_head`` (and divides it), multiple query heads share
    the same K/V head — this is GQA (Llama 2/3, Mistral). MQA is the
    extreme case ``n_kv_head=1``.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        n_kv_head: Optional[int] = None,
    ):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        assert n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = n_head // self.n_kv_head
        self.head_size = n_embd // n_head

        # Separate Q and KV projections so Q has n_head heads and KV has n_kv_head heads.
        self.q_proj = Linear(n_embd, n_head * self.head_size)
        self.kv_proj = Linear(n_embd, 2 * self.n_kv_head * self.head_size)
        self.proj = Linear(n_embd, n_embd)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def __call__(
        self,
        x: Tensor,
        cos: np.ndarray = None,
        sin: np.ndarray = None,
        cache: dict = None,
        layer_idx: int = None,
    ) -> Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)  # (B, T, n_head*hs)
        kv = self.kv_proj(x)  # (B, T, 2*n_kv_head*hs)

        q = q.reshape(B, T, self.n_head, self.head_size).permute(0, 2, 1, 3)
        # kv: (B, T, 2, n_kv_head, hs) -> split along axis 2
        kv = kv.reshape(B, T, 2, self.n_kv_head, self.head_size).permute(2, 0, 3, 1, 4)
        k, v = kv.split(2, axis=0)
        k = k.reshape(B, self.n_kv_head, T, self.head_size)
        v = v.reshape(B, self.n_kv_head, T, self.head_size)

        seq_offset = 0
        if cache is not None and layer_idx in cache:
            seq_offset = cache[layer_idx]["ptr"]

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin, seq_offset=seq_offset)
            k = apply_rope(k, cos, sin, seq_offset=seq_offset)

        if cache is not None:
            # KV-cache is inference-only; building a Tensor from a numpy slice
            # would sever the autograd graph.
            assert not Tensor.requires_grad, (
                "KV-cache is inference-only; wrap cached forward passes in no_grad()"
            )
            if layer_idx not in cache:
                _, nkv, _T_new, hs = k.shape
                max_len = 2048
                cache[layer_idx] = {
                    "k": np.zeros((B, nkv, max_len, hs), dtype=DTYPE),
                    "v": np.zeros((B, nkv, max_len, hs), dtype=DTYPE),
                    "ptr": 0,
                }

            ptr = cache[layer_idx]["ptr"]
            T_new = k.shape[-2]
            cache[layer_idx]["k"][:, :, ptr : ptr + T_new, :] = k.data
            cache[layer_idx]["v"][:, :, ptr : ptr + T_new, :] = v.data
            k_full = Tensor(cache[layer_idx]["k"][:, :, : ptr + T_new, :])
            v_full = Tensor(cache[layer_idx]["v"][:, :, : ptr + T_new, :])
            cache[layer_idx]["ptr"] += T_new
        else:
            k_full, v_full = k, v

        # For GQA: repeat K and V heads so they match Q's head count.
        k_full = repeat_kv(k_full, self.n_rep)
        v_full = repeat_kv(v_full, self.n_rep)

        att = q.matmul(k_full.transpose(-1, -2)) * (1.0 / (self.head_size**0.5))

        # causal mask (only during full-sequence passes, not incremental decoding)
        T_total = k_full.shape[-2]
        if T == T_total:
            mask_data = np.zeros((T, T))
            mask_data[np.triu_indices(T, k=1)] = -1e9
            att = att + Tensor(mask_data)

        att = att.softmax()
        att = self.attn_dropout(att)
        y = att.matmul(v_full)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(y))
