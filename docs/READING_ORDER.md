# Reading Order

If you've never worked on a neural network from scratch before, this is
the order in which to read the code. Each file builds directly on the
ones above it. Read them like chapters.

## 1. `src/llm/tensor.py` - the autograd engine
**Read this first.** Nothing else makes sense without it. This file
implements a minimal reverse-mode automatic differentiation system on
top of NumPy. The only class you need to understand is `Tensor`:

- `Tensor.data` holds the actual numbers (a NumPy array).
- `Tensor.grad` holds the gradient, same shape as `data`.
- `Tensor._prev` is the list of parent tensors - the inputs to the
  operation that produced this tensor.
- `Tensor._backward` is a closure that, when called, adds this node's
  contribution to its parents' gradients.

Start at `__add__` - it's the simplest op with a backward. Then
`__mul__`, `matmul`, `sum`, and finally `backward()`, which does a
topological sort and walks the graph in reverse to apply the chain rule.

Also note `no_grad()` (disables graph building for inference) and
`checkpoint()` (trades compute for memory by recomputing forward in
backward - come back to this one after you've read `transformer.py`).

## 2. `src/llm/nn.py` - layer building blocks
Now that you can differentiate, you need layers. In order:

- `Module` - base class with auto-discovered parameters. All layers
  inherit from this.
- `Linear` - the most basic layer: `y = xW + b`.
- `Embedding` - a lookup table mapping token IDs to vectors. This is
  how text enters the model.
- `LayerNorm` / `RMSNorm` - stabilize activations across the feature
  axis. RMSNorm is the modern default (used by Llama and Mistral).
- `SiLU`, `GELU`, `ReLU`, `Dropout` - activation functions and
  dropout. SiLU (also called Swish) is what SwiGLU is built from.

## 3. `src/llm/data.py` - tokenization and batching
You need to turn text into integers the model can consume.

- `Tokenizer` - a trivial character tokenizer. Good for learning.
- `BPETokenizer` - byte-pair encoding, the same family of tokenizer
  GPT, Llama, and most real LLMs use.
- Both tokenizers support `to_dict()` / `from_dict()` for embedding
  inside model checkpoints.
- `Dataset` - slices a long token stream into `(x, y)` pairs where `y`
  is `x` shifted by one. This is the next-token-prediction objective.

## 4. `src/llm/attention.py` - the attention mechanism
The heart of the transformer.

- `precompute_rope_freqs` + `apply_rope` - Rotary Position
  Embeddings. Read `precompute` first, then the rotation math in
  `apply_rope`.
- `MultiHeadAttention` - scaled dot-product attention, split into
  multiple heads. Supports Grouped-Query Attention (`n_kv_head < n_head`)
  the way Llama 2/3 does.
- `repeat_kv` - tiny helper for GQA that duplicates K/V heads so they
  match the Q head count.

Key lines to study: the causal mask (preventing the model from
"cheating" by looking ahead), and the KV-cache branch (inference-only
shortcut used during generation).

## 5. `src/llm/transformer.py` - assembling the model
With attention in hand, the rest is straightforward:

- `FeedForward` - two (or three, for SwiGLU) linear layers with a
  nonlinearity in between.
- `Block` - one transformer layer: `x = x + attn(norm(x))`, then
  `x = x + ffn(norm(x))`. Pre-norm residual architecture.
- `Transformer` - stacks N blocks, adds token embedding at the input
  and an LM head at the output. Optional weight tying.
- `generate` - inference: sample tokens one at a time, optionally
  using a KV cache for speed.

Also note the optional `grad_checkpoint` flag in the `__call__` - this
uses `tensor.checkpoint` to trade compute for memory by recomputing
each block in the backward pass.

## 6. `src/llm/train.py` - the training loop
- `cross_entropy` - numerically stable loss via log-sum-exp.
- `Adam` - the optimizer. Read the update rule carefully; it's less
  mysterious than it looks.
- `estimate_loss` / `perplexity` - validation metrics. Perplexity is
  just `exp(cross_entropy)` and is what LLM papers report.
- `train_loop` - batch sampling, forward pass, backward pass,
  gradient clipping, optimizer step, LR schedule, checkpointing.

## 7. `src/llm/scheduler.py` - learning-rate schedule
Linear warmup followed by cosine decay. Twenty lines. Read it when
`train_loop` references it.

## 8. `src/llm/config.py` - hyperparameters in one place
`ModelConfig` and `TrainConfig` dataclasses. You'll see them everywhere.

## 9. `src/llm/cli.py` - command-line entry point
Glues everything together for `llm train` and `llm generate`.
Tokenizers are cached on disk (keyed by data file, type, and vocab
size) and embedded in checkpoints, so `generate` only needs a
checkpoint path and a prompt. Useful as an example of how to build a
real training script, but skippable.

## 10. `tests/` - runnable examples
Each test file mirrors one of the source files. Good place to copy
snippets from when you're learning an API.

## 11. `docs/BLUEPRINT.md` - implement it in a different language
Once you've read the code, `BLUEPRINT.md` is the language-agnostic
spec that describes the same system in math and pseudocode. Use it
as a reference if you want to rewrite this project in Rust or Julia
or whatever.

## 12. `scripts/profile.py` - where does the time actually go?
Run `python scripts/profile.py` and you'll see the per-component
time breakdown. Attention and FFN dominate; everything else is noise.
This is a fact worth internalizing. All model and runtime parameters
are configurable via CLI flags (`--n_embd`, `--n_layer`, `--batch`,
`--seq_len`, etc.). You can also pass `--checkpoint path/to/model.npz`
to profile a trained model with its real weights and architecture.