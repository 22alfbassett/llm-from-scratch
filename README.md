# llm-from-scratch

A small decoder-only transformer language model built from the ground up on top of NumPy (with optional CuPy for GPU). No PyTorch, no JAX - just a hand-written `Tensor` with reverse-mode autograd, a tiny `nn` module library, and a training loop.

The goal is pedagogical: every piece (tensor ops, backprop, attention, RoPE, KV-cache, SwiGLU, RMSNorm, BPE tokenizer, Adam, cosine schedule) is visible and editable in a few hundred lines.

## Features

- Reverse-mode autograd `Tensor` with broadcasting, matmul, softmax, reshape, split, etc.
- Decoder-only Transformer with:
  - Multi-head self-attention, causal mask, KV-cache for generation
  - Rotary position embeddings (RoPE)
  - RMSNorm or LayerNorm (configurable)
  - SwiGLU / GELU / ReLU FFN (configurable)
  - Weight tying between token embedding and output head
- BPE and char-level tokenizers
- Adam with weight decay, gradient clipping, cosine LR schedule with warmup, gradient accumulation, validation loss
- Checkpoints embed their `ModelConfig` and tokenizer, so `Transformer.from_checkpoint(path)` reconstructs the exact architecture and tokenizer in one call
- Runs on CPU (NumPy) or GPU (CuPy) - select via `LLM_DEVICE=cpu|gpu`

## Layout

```
src/llm/          package source
  tensor.py         autograd Tensor, no_grad(), set_seed()
  nn.py             Module, Linear, Embedding, LayerNorm, RMSNorm, ...
  attention.py      RoPE + MultiHeadAttention
  transformer.py    FeedForward, Block, Transformer
  data.py           Tokenizer, BPETokenizer, Dataset
  train.py          cross_entropy, Adam, train_loop, estimate_loss
  scheduler.py      CosineWithWarmup
  config.py         ModelConfig, TrainConfig dataclasses
  cli.py            command-line entry point
tests/            pytest suite (autograd gradcheck, model, train smoke test)
data/             training corpora + tokenizers/ (cached by name_type_vocabsize)
scripts/          plot_results.py, run_experiments.py
docs/             design notes, guides, roadmap
checkpoints/      saved .npz models with embedded tokenizer (gitignored)
results/          per-run JSON metrics (gitignored)
```

## Install

```bash
pip install -e .            # CPU
pip install -e '.[gpu]'     # + CuPy for NVIDIA GPUs
pip install -e '.[dev]'     # + pytest, ruff
```

## Quick start

Train a tiny Shakespeare model:

```bash
python -m llm.cli train --dataset shakespeare --n_layer 4 --n_embd 128 --n_head 4 --steps 500
```

Generate text using said model:

```bash
python -m llm.cli generate --checkpoint checkpoints/shakespeare_bpe_L4_E128_H4.npz --prompt "The "
```

The tokenizer is embedded in the checkpoint, so generation only needs the checkpoint file and a prompt.

Training reads `data/{dataset}.txt`, trains or loads a cached BPE tokenizer from `data/tokenizers/`, writes the checkpoint (with embedded tokenizer) to `checkpoints/`, and dumps a JSON metrics file into `results/`. Tokenizers are cached as `data/tokenizers/{dataset}_{type}_{vocab_size}.json` so the same tokenizer is reused across training runs with matching settings. Use `--name` to override the checkpoint name (defaults to the dataset name).

Resume an interrupted run (restores optimizer state so training picks up where it left off):

```bash
python -m llm.cli train --dataset shakespeare --resume checkpoints/shakespeare_bpe_L4_E128_H4.npz --steps 5000
```

Fine-tune on a new corpus (fresh optimizer, uses the model and tokenizer from the checkpoint):

```bash
python -m llm.cli train --dataset new_corpus --from_checkpoint checkpoints/shakespeare_bpe_L4_E128_H4.npz --steps 500
```

Force CPU:

```bash
LLM_DEVICE=cpu python -m llm.cli ...
```

## Programmatic API

```python
from llm import ModelConfig, Transformer, set_seed
from llm.data import BPETokenizer, Dataset
from llm.train import train_loop

set_seed(0)

# Train a tokenizer
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=1000)

cfg = ModelConfig(vocab_size=tokenizer.vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64)
model = Transformer(cfg)

tokens = tokenizer.encode(text)
ds = Dataset(tokens, block_size=cfg.block_size)
history = train_loop(model, ds, steps=1000, lr=3e-4, batch_size=16, tokenizer=tokenizer)

# Save with embedded tokenizer
model.save("checkpoints/demo.npz", tokenizer=tokenizer)

# Load — returns (model, tokenizer), everything needed to generate
model2, tok2 = Transformer.from_checkpoint("checkpoints/demo.npz")

prompt_ids = tok2.encode("The ")
for token_id in model2.generate(prompt_ids, max_new_tokens=100, temperature=0.8):
    print(tok2.decode([token_id]), end="")
```

## Tests

```bash
PYTHONPATH=src LLM_DEVICE=cpu python -m pytest tests/
```

The suite includes finite-difference gradient checks for the autograd ops, a causal-mask leakage test, parametrized model-variant tests (swiglu/gelu/relu x rmsnorm/layernorm x tied/untied), a checkpoint roundtrip, and a loss-decreases smoke test on a structured tiny corpus.

## Status

This is a research/learning project, expect rough edges. Checkpoints and training results are not committed; retrain from scratch.
