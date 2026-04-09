"""Where does the time actually go in a transformer forward pass?

Runs a model and reports per-component runtime breakdown, parameter counts
per layer type, memory estimates, and throughput.

Usage:
    PYTHONPATH=src python scripts/profile.py
    PYTHONPATH=src python scripts/profile.py --n_embd 512 --n_head 8 --n_layer 12
    PYTHONPATH=src python scripts/profile.py --checkpoint results/my_model.npz
    PYTHONPATH=src python scripts/profile.py --batch 1 --seq_len 64 --iters 50
"""

import argparse
import time

import numpy as np

from llm.config import ModelConfig
from llm.tensor import DTYPE, Tensor, no_grad
from llm.transformer import Transformer


def _time(fn, n: int = 20) -> float:
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0  # ms / iter


def _param_count(module) -> int:
    return sum(p.data.size for p in module.parameters())


def _param_bytes(module) -> int:
    return sum(p.data.nbytes for p in module.parameters())


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KB"
    return f"{n} B"


def _activation_bytes(B: int, T: int, cfg: ModelConfig) -> int:
    """Rough estimate of peak activation memory for a forward pass."""
    d = cfg.n_embd
    V = cfg.vocab_size
    L = cfg.n_layer
    # per-layer: attention scores + QKV + FFN intermediates
    attn_scores = B * cfg.n_head * T * T  # attention weight matrix
    qkv = 3 * B * T * d  # Q, K, V projections
    ffn_hidden = B * T * (d * cfg.ffn_multiplier)  # FFN intermediate
    per_layer = (attn_scores + qkv + ffn_hidden) * np.dtype(DTYPE).itemsize
    # logits
    logits = B * T * V * np.dtype(DTYPE).itemsize
    return per_layer * L + logits


def main():
    p = argparse.ArgumentParser(description="Profile transformer components")
    # model params
    p.add_argument("--vocab_size", type=int, default=1000)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--act", type=str, default="swiglu", choices=["swiglu", "gelu", "relu"])
    p.add_argument("--norm", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"])
    p.add_argument("--ffn_multiplier", type=int, default=4)
    p.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.npz)")
    # runtime params
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=None, help="defaults to block_size")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model, _tokenizer = Transformer.from_checkpoint(args.checkpoint)
        cfg = model.config
    else:
        cfg = ModelConfig(
            vocab_size=args.vocab_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            block_size=args.block_size,
            dropout=0.0,
            act=args.act,
            norm=args.norm,
            ffn_multiplier=args.ffn_multiplier,
            n_kv_head=args.n_kv_head,
        )
        model = Transformer(cfg)

    B = args.batch
    T = args.seq_len or cfg.block_size
    if T > cfg.block_size:
        print(f"warning: seq_len ({T}) > block_size ({cfg.block_size}), clamping")
        T = cfg.block_size
    iters = args.iters

    x_ids = [[i % cfg.vocab_size for i in range(T)] for _ in range(B)]
    x_emb = Tensor.randn(B, T, cfg.n_embd)

    # Extract components for fine-grained profiling
    emb = model.token_embedding
    block = model.blocks[0]
    mha = block.sa
    ffn = block.ffwd

    model.eval()

    # ── Timing ──
    with no_grad():
        t_emb = _time(lambda: emb(x_ids), iters)
        t_mha = _time(lambda: mha(x_emb), iters)
        t_ffn = _time(lambda: ffn(x_emb), iters)
        t_block = _time(lambda: block(x_emb), iters)
        t_full = _time(lambda: model(x_ids), iters)

    # ── Parameter breakdown ──
    p_emb = _param_count(emb)
    p_mha = _param_count(mha)
    p_ffn = _param_count(ffn)
    p_block = _param_count(block)
    p_total = _param_count(model)

    b_total = _param_bytes(model)

    tokens_per_sec = (B * T) / (t_full / 1000.0)
    act_mem = _activation_bytes(B, T, cfg)

    # ── Display ──
    W = 60
    print("=" * W)
    print("MODEL CONFIGURATION")
    print("=" * W)
    print(f"  vocab_size:     {cfg.vocab_size:,}")
    print(f"  n_embd:         {cfg.n_embd}")
    print(f"  n_head:         {cfg.n_head}")
    n_kv_display = cfg.n_kv_head or cfg.n_head
    gqa_ratio = cfg.n_head // n_kv_display
    gqa_label = "MHA" if gqa_ratio == 1 else f"GQA (ratio {gqa_ratio}:1)"
    print(f"  n_kv_head:      {n_kv_display}  [{gqa_label}]")
    print(f"  n_layer:        {cfg.n_layer}")
    print(f"  block_size:     {cfg.block_size}")
    print(f"  activation:     {cfg.act}")
    print(f"  norm:           {cfg.norm}")
    print(f"  ffn_multiplier: {cfg.ffn_multiplier}")
    print(f"  dtype:          {DTYPE}")

    print()
    print("=" * W)
    print("PARAMETERS")
    print("=" * W)
    print(f"  {'component':<28}{'params':>14}{'share':>10}")
    print(f"  {'-' * 52}")
    param_rows = [
        ("embedding", p_emb),
        ("attention (per layer)", p_mha),
        ("ffn (per layer)", p_ffn),
        ("block (per layer)", p_block),
        (f"all blocks ({cfg.n_layer} layers)", p_block * cfg.n_layer),
        ("total model", p_total),
    ]
    for name, count in param_rows:
        share = f"{100 * count / p_total:.1f}%" if p_total else "n/a"
        print(f"  {name:<28}{count:>14,}{share:>10}")

    print()
    print(f"  model weight memory: {_fmt_bytes(b_total)}")
    print(f"  est. activation memory (fwd): {_fmt_bytes(act_mem)}")
    print(f"  est. total (weights + act): {_fmt_bytes(b_total + act_mem)}")

    print()
    print("=" * W)
    print(f"TIMING  (batch={B}, seq_len={T}, iters={iters})")
    print("=" * W)
    print(f"  {'component':<28}{'ms/iter':>12}{'share':>10}")
    print(f"  {'-' * 50}")
    timing_rows = [
        ("embedding", t_emb),
        ("attention (1 layer)", t_mha),
        ("ffn (1 layer)", t_ffn),
        ("full block (1 layer)", t_block),
        ("full forward", t_full),
    ]
    for name, t in timing_rows:
        share = f"{100 * t / t_full:.1f}%" if t_full > 0 else "n/a"
        print(f"  {name:<28}{t:>12.3f}{share:>10}")

    print()
    overhead = t_full - t_block * cfg.n_layer
    print(f"  n_layer * block = {t_block * cfg.n_layer:.3f} ms")
    print(f"  full forward    = {t_full:.3f} ms")
    print(f"  overhead (emb + norm + head) = {overhead:.3f} ms")
    print(f"  throughput: {tokens_per_sec:,.0f} tokens/sec")

    print()
    print("=" * W)
    print("SCALING ESTIMATE")
    print("=" * W)
    for mult in [2, 4, 8]:
        scaled_params = p_total * mult
        # very rough: runtime scales ~linearly with param count for fixed shape
        print(f"  {mult}x params (~{scaled_params:,}): ~{t_full * mult:.1f} ms/fwd")

    print("=" * W)


if __name__ == "__main__":
    main()
