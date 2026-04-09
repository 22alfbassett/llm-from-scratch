# LLM from Scratch: A Language-Agnostic Blueprint

This document is a complete, step-by-step specification for building a
modern decoder-only Transformer language model from first principles.
It is designed so that someone with **no prior machine learning
experience** can read it top to bottom and end up with a working LLM in
the language of their choice.

The reference implementation is Python + NumPy, but nothing in this
document is Python-specific. Every algorithm is described in math and
pseudocode.

## How to read this document

- **Read in order.** Each milestone depends on the previous ones.
  Skipping ahead will not work - this is the shortest path through the
  material, not a menu.
- **Each milestone has the same structure:**
  1. **What and why** - a plain-English explanation of what you're
     building and why it exists.
  2. **Math** - the equations you will implement.
  3. **Function signatures** - every function or class you need to
     write, with types and a one-line description.
  4. **Pseudocode** - the actual algorithm, in language-agnostic form.
  5. **Checkpoints** - small tests you can run to verify that this
     milestone works before moving on.
  6. **Common pitfalls** - mistakes the author has made so you don't
     have to.
- **You do not need to understand everything the first time.** Come
  back to this document after reading the code.

## Prerequisites

You need:
- A programming language with **arrays / tensors**. NumPy, PyTorch,
  JAX, ndarray (Rust), Array (Julia), nalgebra (Rust), Eigen (C++),
  or any equivalent.
- **Element-wise operations, matrix multiplication, and basic
  reductions** (sum, mean, max) on multi-dimensional arrays.
- Comfort with **basic calculus**: you should know what a derivative
  is, what the chain rule says, and roughly what a gradient is. You
  do not need to have taken a course.
- **Linear algebra at the level of "I know what matrix multiplication
  is and what shapes fit together."** Nothing more.

You do *not* need:
- Any prior ML knowledge.
- A GPU.
- A deep learning framework. You will build your own.

---

# Milestone 0: Mental model

Before writing any code, understand what you're building.

A **language model** is a function that takes a sequence of tokens
(numbers representing pieces of text) and returns, for each position,
a probability distribution over "what token comes next." Training
means: show the model billions of (input, next-token) pairs from real
text and nudge its parameters so it gets better at predicting.

A **Transformer** is one particular architecture for building such a
function. It is made of stacks of two things: **attention** (lets each
token look at other tokens) and **feed-forward networks** (lets each
token do computation on its own representation).

To train it, you need:

1. A way to compute **how wrong** the model is (a **loss function**).
2. A way to compute, for every parameter in the model, **how much
   that parameter contributed to the wrongness** (a **gradient**).
3. A way to use those gradients to **update the parameters** (an
   **optimizer**).

Computing gradients by hand is impossible for anything larger than a
toy model. So your first task is to build a system that does it
automatically - **automatic differentiation**, a.k.a. **autograd**.

---

# Milestone 1: The autograd engine

**What and why.** You're building the mathematical equivalent of a
tape recorder: a system that watches every arithmetic operation, builds
a graph of how values depend on each other, and can then play the
graph back in reverse to compute derivatives. Every modern ML
framework (PyTorch, JAX, TensorFlow) has one of these at its core.

## 1.1 The Tensor class

**Core idea.** A `Tensor` is a multi-dimensional array of floating-point
numbers *with memory*. It remembers which operation produced it and
which tensors were the inputs, so it can later ask those inputs to
contribute to its gradient.

**Fields:**

| Field         | Type           | Purpose                                                           |
| :------------ | :------------- | :---------------------------------------------------------------- |
| `data`        | N-D float array| The actual numbers.                                               |
| `grad`        | N-D float array| The derivative of some scalar loss w.r.t. this tensor. Same shape as `data`. Starts at zero. |
| `_prev`       | set of Tensors | Parents: the tensors that were inputs to the op that made this. |
| `_op`         | string         | Name of the op (debugging only).                                  |
| `_backward`   | closure        | Function that, when called, adds this node's contribution to each parent's `grad`. |

**Function signatures:**

```
Tensor(data, _prev=(), _op='')   # constructor
Tensor.zeros(*shape) -> Tensor
Tensor.randn(*shape) -> Tensor   # standard normal samples

# Operations you need (each returns a new Tensor with a _backward):
Tensor + Tensor -> Tensor
Tensor * Tensor -> Tensor        # element-wise
Tensor ** float -> Tensor        # element-wise power
Tensor.matmul(Tensor) -> Tensor  # matrix multiplication
Tensor.sum(axis=None) -> Tensor
Tensor.mean(axis=-1) -> Tensor
Tensor.exp() -> Tensor
Tensor.tanh() -> Tensor
Tensor.sigmoid() -> Tensor
Tensor.relu() -> Tensor
Tensor.softmax() -> Tensor       # along last axis
Tensor.reshape(*shape) -> Tensor
Tensor.transpose(a, b) -> Tensor
Tensor.permute(*axes) -> Tensor
Tensor.split(n, axis) -> list[Tensor]

Tensor.backward(grad=None) -> None   # entry point for reverse-mode AD
```

## 1.2 The chain rule, operationally

For every operation $z = f(x, y)$, you implement:

1. The **forward pass**: compute `z.data` from `x.data` and `y.data`.
2. The **backward pass**: given $\frac{\partial L}{\partial z}$
   (available as `z.grad` when backward time comes), add to each parent:
   - $\frac{\partial L}{\partial x} \mathrel{+}= \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$
   - $\frac{\partial L}{\partial y} \mathrel{+}= \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y}$

**Accumulate (`+=`)** - never overwrite. A tensor can be used multiple
times in the graph, and each use must contribute to its gradient.

## 1.3 Local gradients for every op you need

| Operation             | Forward                              | Backward (local partial)                                                    |
| :-------------------- | :----------------------------------- | :-------------------------------------------------------------------------- |
| Add $z=x+y$           | elementwise                          | $dx = dz$, $dy = dz$                                                        |
| Multiply $z=x\cdot y$ | elementwise                          | $dx = y \cdot dz$, $dy = x \cdot dz$                                        |
| Power $z=x^n$         | elementwise                          | $dx = n \cdot x^{n-1} \cdot dz$                                             |
| Matmul $Z=XW$         | $Z_{ij}=\sum_k X_{ik}W_{kj}$         | $dX = dZ \cdot W^\top$, $dW = X^\top \cdot dZ$                              |
| Sum over axis         | $z = \sum_i x_i$                     | $dx_i = dz$ (broadcast)                                                     |
| Mean over axis        | $z = \frac{1}{N}\sum_i x_i$          | $dx_i = \frac{1}{N} dz$                                                     |
| ReLU                  | $\max(0, x)$                         | $dx = \mathbb{1}[x>0] \cdot dz$                                             |
| Sigmoid $\sigma$      | $\frac{1}{1+e^{-x}}$                 | $dx = \sigma(x)(1-\sigma(x)) \cdot dz$                                      |
| Tanh                  | $\tanh(x)$                           | $dx = (1 - \tanh^2(x)) \cdot dz$                                            |
| Exp                   | $e^x$                                | $dx = e^x \cdot dz$                                                         |
| Softmax $P_i$         | $\frac{e^{x_i}}{\sum_j e^{x_j}}$     | $dx_i = P_i \cdot (dz_i - \sum_j P_j dz_j)$                                 |
| Reshape/transpose     | permute indices                      | inverse permutation of `dz`                                                 |

## 1.4 Broadcasting

If you add a tensor of shape $(1, D)$ to one of shape $(B, D)$, the
smaller one is implicitly replicated along the batch axis during the
forward pass. In the backward pass, the gradient flowing back to the
smaller tensor must be **summed** over the replicated axis to match
the original shape. Write a helper:

```
reduce_grad(grad, target_shape)
    while grad has more dims than target_shape:
        grad = grad.sum(axis=0)
    for each axis i where target_shape[i] == 1 and grad.shape[i] > 1:
        grad = grad.sum(axis=i, keepdims=True)
    return grad
```

Call this in every `_backward` for ops whose inputs could be broadcast.

## 1.5 The backward traversal

```
function Tensor.backward(self, grad=None):
    # 1. topological sort of the graph rooted at self
    topo = empty list
    visited = empty set
    function visit(node):
        if node in visited: return
        visited.add(node)
        for parent in node._prev:
            visit(parent)
        topo.append(node)
    visit(self)

    # 2. zero all gradients in the subgraph
    for node in topo:
        node.grad = zeros_like(node.data)

    # 3. seed the root
    if grad is not None:
        self.grad = grad
    else:
        self.grad = ones_like(self.data)  # dL/dL = 1

    # 4. walk in reverse order, applying each local backward
    for node in reverse(topo):
        node._backward()
```

**Why topological order matters.** A node's gradient must be fully
accumulated (from all downstream uses) *before* you call its `_backward`
- otherwise you'd propagate a partial gradient and get the wrong answer.

## 1.6 `no_grad` and `checkpoint` (come back to these later)

Add a thread-local or global flag `requires_grad` that defaults to `True`.
When it is `False`, operations skip building the graph (`_prev` stays
empty, `_backward` is a no-op). Wrap this in a `no_grad()` context
manager. You'll use it during inference and evaluation.

**Gradient checkpointing** (optional, come back after Milestone 5) is
a memory-saving trick. Given a function `fn(x)`:

1. Run `fn(x)` once with `requires_grad=False` - you get the output
   data but do not keep the interior graph.
2. Return a new Tensor whose `_backward` will, when called, re-run
   `fn(x')` with `requires_grad=True` on *copies* of the inputs, call
   `.backward(grad=out.grad)` on that local graph, and add the resulting
   gradients back into the originals.

This trades extra forward computation for lower peak memory, which
matters when training deep models.

## 1.7 Checkpoints - verify Milestone 1

Write these as unit tests. **Do not proceed until they pass.**

1. **Scalar chain rule:** Let $y = x^3$, $x = 2$. Then $\frac{dy}{dx} = 3x^2 = 12$. Your autograd should return `x.grad == 12`.
2. **Matmul gradient:** Let $Y = X W$, sum all elements. Then
   $\frac{\partial (\sum Y)}{\partial X} = \mathbf{1}_{Y} W^\top$. Check against this formula.
3. **Finite-difference gradcheck:** For any differentiable $f$, compare
   your analytic gradient against $\frac{f(x+h) - f(x-h)}{2h}$ with
   $h = 10^{-4}$ and assert they agree to $\sim 10^{-3}$.
4. **Broadcasting:** $(1, D) + (B, D)$ followed by `.sum()` - the
   gradient flowing to the smaller tensor must have shape $(1, D)$.

## 1.8 Common pitfalls

- **Forgetting `+=` in backward.** If you use `=` anywhere, a shared
  subgraph will silently overwrite gradients.
- **Mutating `data` in place.** Don't. Create new tensors.
- **Not zeroing gradients before a new backward pass.** The optimizer
  expects the current step's gradient, not an accumulation from all
  previous steps.
- **Float precision.** Use `float32` (or higher) for any reduction. If
  you use `float16` storage, promote to `float32` inside softmax,
  normalization, and loss calculations.

---

# Milestone 2: Neural network building blocks

**What and why.** Now that you can differentiate arbitrary expressions,
you need a clean way to define *layers* that own parameters (weights)
and can be stacked. This is the `Module` abstraction that every ML
framework has.

## 2.1 The `Module` base class

**Purpose.** A `Module` is an object that owns Tensor parameters and
optionally contains sub-modules. The base class provides:

- `parameters()` - yields every Tensor parameter recursively.
- `zero_grad()` - zeros every parameter's `grad`.
- `train()` / `eval()` - flips a `training` flag (used by Dropout).

**Trick.** You can auto-discover parameters by scanning the object's
attributes: anything that is a `Tensor` is a parameter; anything that
is a `Module` contributes its own parameters; anything that is a list
or tuple, recurse into it. No explicit registration needed.

```
class Module:
    training = True

    def parameters(self):
        seen = set()
        for value in self.__dict__.values():
            yield from _visit(value, seen)

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)

    def train(self):  self.training = True;  recurse into submodules
    def eval(self):   self.training = False; recurse into submodules
```

## 2.2 Linear layer

**Math.** $y = xW + b$, where $W$ has shape $(n_{\text{in}}, n_{\text{out}})$ and $b$ has shape $(1, n_{\text{out}})$.

**Init.** Set $W \sim \mathcal{N}(0, 1/\sqrt{n_{\text{in}}})$ (Xavier/Glorot). Set $b = 0$.

**Why small init?** Large random weights cause the forward activations
to explode layer by layer. Xavier keeps the variance of activations
roughly constant across layers.

```
class Linear(Module):
    def __init__(self, n_in, n_out):
        self.weight = randn(n_in, n_out) * (1 / sqrt(n_in))
        self.bias   = zeros(1, n_out)

    def __call__(self, x):
        return x.matmul(self.weight) + self.bias
```

## 2.3 Embedding layer

**Purpose.** Map integer token IDs to dense vectors. This is how
discrete text enters a model that only understands continuous math.

**Implementation.** A lookup table: a weight matrix of shape $(V, d)$.
For input IDs of shape $(B, T)$, the output is the rows of the weight
matrix indexed by those IDs - shape $(B, T, d)$.

**Backward.** The gradient for each output position must be *added*
(not assigned) to the corresponding row of the weight matrix's
gradient. In NumPy this is `np.add.at(weight.grad, ids, out.grad)` -
you need an atomic scatter-add because the same token can appear
multiple times.

## 2.4 LayerNorm

**What.** For each position independently, subtract the mean and divide
by the standard deviation across the feature axis, then apply a learned
scale and shift.

**Math.** For feature vector $x$ of dimension $d$:

$$\mu = \frac{1}{d}\sum_i x_i, \quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$$

where $\gamma, \beta \in \mathbb{R}^d$ are learned, initialized to $1$ and $0$.

## 2.5 RMSNorm

**What.** Like LayerNorm but drops the mean subtraction and the bias.
Used by Llama, Mistral, and most modern LLMs.

$$\text{rms}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}, \quad y_i = \gamma_i \cdot \frac{x_i}{\text{rms}(x)}$$

**Why prefer RMSNorm?** Cheaper (one reduction instead of two, one
parameter vector instead of two) and empirically just as effective.

## 2.6 Activation functions

| Name   | Formula                                  | Where used                         |
| :----- | :--------------------------------------- | :--------------------------------- |
| ReLU   | $\max(0, x)$                             | Legacy. Simple, prone to "dead neurons." |
| GELU   | $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$ | BERT, GPT-2.                          |
| SiLU   | $x \cdot \sigma(x)$                      | Building block of SwiGLU.          |

## 2.7 Dropout

**What.** During training only, randomly zero out a fraction $p$ of
activations and scale the rest by $\frac{1}{1-p}$ (so the expected
activation is unchanged). During eval, it's a no-op.

**Why.** Prevents co-adaptation of features; acts as a regularizer.

## 2.8 Checkpoints - verify Milestone 2

1. A `Linear(4, 8)` fed a $(2, 4)$ input should return shape $(2, 8)$.
2. After `module.zero_grad()`, every parameter's gradient is all zeros.
3. `module.parameters()` must return every `Tensor` inside the module
   and its sub-modules, with no duplicates.
4. A `LayerNorm(d)` applied to random input, then summed and
   backpropped, should produce finite, non-NaN gradients.

## 2.9 Common pitfalls

- **Zero init for Linear weights.** Breaks symmetry: every neuron
  learns the same thing. Always use small random init.
- **Forgetting the training/eval flag in Dropout.** If you dropout
  during evaluation, your results will be noisy and wrong.
- **Using `float16` inside LayerNorm/RMSNorm.** The variance can
  underflow. Always promote to at least `float32`.

---

# Milestone 3: Tokenization and data

**What and why.** Models operate on integer IDs, not characters or
bytes directly. A **tokenizer** converts text ↔ integers. A **dataset**
chops long token streams into fixed-length training examples.

## 3.1 Character tokenizer (simplest, start here)

```
class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(unique(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):  return [self.stoi[ch] for ch in s]
    def decode(self, ids): return ''.join(self.itos[i] for i in ids)
```

Good enough to get you training. Limitation: vocab is tied to the
characters you happen to see; cannot handle unseen characters.

## 3.2 BPE (Byte-Pair Encoding)

**What.** Start with a vocabulary of 256 tokens (one per byte).
Repeatedly: find the most frequent *pair* of adjacent tokens in the
training corpus, create a new token for that pair, and replace all
occurrences. Stop when the vocab reaches the target size.

**Why.** Handles any UTF-8 string (including languages and characters
you've never seen), while giving common words/subwords dedicated
short tokens. This is what GPT, Llama, and Mistral use.

```
class BPETokenizer:
    def train(self, text, vocab_size):
        tokens = list(text.encode('utf-8'))
        merges = {}   # (pair) -> new_token_id
        vocab = {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            stats = count pairs in tokens
            if no pairs: break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            tokens = replace all pair in tokens with new_id
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        self.merges, self.vocab = merges, vocab

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        # apply merges in the order they were learned
        for (p0, p1), new_id in sorted(self.merges.items(), by value):
            tokens = replace all (p0, p1) with new_id in tokens
        return tokens

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')
```

## 3.3 Dataset and batching

**Purpose.** Given a long stream of token IDs, produce training batches
of shape $(B, T)$ for inputs and $(B, T)$ for targets, where target is
the input shifted by one position.

```
class Dataset:
    def __init__(self, data, block_size):
        self.data = array(data, dtype=uint32)
        self.block_size = block_size

    def get_batch(self, batch_size):
        ix = random integers in [0, len(data) - block_size - 1), count = batch_size
        x = stack([data[i : i+block_size]     for i in ix])
        y = stack([data[i+1 : i+1+block_size] for i in ix])
        return x, y
```

**Why shift by one?** Each position's label is "the token that actually
came next in the text." The model learns to predict that at every
position in parallel - this is what makes transformer training
efficient.

## 3.4 Checkpoints - verify Milestone 3

1. `encode(decode(ids)) == ids` for any list of IDs in the vocab.
2. `decode(encode(text)) == text` for any text used during training.
3. `Dataset.get_batch(B)` returns `(x, y)` of the right shapes, and
   `y[:, :-1] == x[:, 1:]` (the shift invariant).

---

# Milestone 4: Attention

**What and why.** Attention is what makes Transformers work. It lets
each token in a sequence look at every previous token and pull in
information from them. The "self" in self-attention means: the queries,
keys, and values all come from the same sequence.

## 4.1 Scaled dot-product attention (single head)

Given input $X$ of shape $(B, T, d)$, project into three separate
spaces via learned linear layers:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Compute the attention scores:

$$A = \frac{QK^\top}{\sqrt{d}}$$

**Why divide by $\sqrt{d}$?** Without it, as $d$ grows the dot products
grow too, pushing softmax into regions where gradients are nearly zero.
The scale factor keeps things stable.

Apply a **causal mask** - for each position $i$, set $A_{ij} = -\infty$
for $j > i$. This prevents "cheating" (a model that could peek at
future tokens would trivially win at next-token prediction and learn
nothing useful).

Convert to probabilities and compute the weighted sum:

$$P = \text{softmax}(A), \quad \text{output} = PV$$

## 4.2 Multi-head attention

**Why multiple heads?** One attention head can only learn one
"relation." Splitting $d$ into $h$ heads of dimension $d/h$ and
running attention independently in each lets the model learn different
kinds of relations simultaneously.

```
reshape Q, K, V from (B, T, d) to (B, h, T, d/h)
compute attention per head (in parallel via batched matmul)
reshape back to (B, T, d)
apply final output projection W_O of shape (d, d)
```

## 4.3 Grouped-Query Attention (GQA)

**What.** A modern refinement: use $h_q$ query heads but only $h_{kv}
< h_q$ key/value heads. Query heads within a group share the same K
and V. Set $h_{kv} = 1$ and you get Multi-Query Attention (MQA); set
$h_{kv} = h_q$ and you get standard MHA.

**Why.** The KV cache during inference is dominated by K and V
storage; fewer KV heads = much smaller cache = faster generation
with essentially no quality loss. Used by Llama 2/3, Mistral.

**Implementation.** Compute $Q$ of shape $(B, h_q, T, d/h_q)$ and $K,
V$ of shape $(B, h_{kv}, T, d/h_q)$. Before computing $QK^\top$,
**repeat** K and V along the head axis so they match $h_q$ - each
$K_{\text{kv-head}}$ is duplicated $h_q / h_{kv}$ times. The backward
pass sums the gradients from the duplicated heads back into the
original head.

```
function repeat_kv(x, n_rep):
    # x shape: (B, n_kv, T, d); output (B, n_kv * n_rep, T, d)
    if n_rep == 1: return x
    return x repeated along the head axis with n_rep copies per original head
```

## 4.4 Rotary Position Embeddings (RoPE)

**What.** The transformer core is **permutation-invariant** - if you
shuffle the tokens, the output shuffles with them. You need a way to
tell the model where each token sits. RoPE does this by *rotating*
pairs of dimensions in Q and K by an angle that depends on position.

**Why rotation?** The key property: after rotation, the dot product
$Q_m \cdot K_n$ depends only on the *relative* offset $m - n$, not the
absolute positions. This gives the model a natural notion of "this
token is $k$ steps after that one" and helps it extrapolate to lengths
beyond those seen during training.

**Math.** For each pair of adjacent dimensions $(x_{2i}, x_{2i+1})$ at
position $m$:

$$\theta_i = 10000^{-2i/d}$$

$$\begin{pmatrix} y_{2i} \\ y_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Precompute the $\cos$ and $\sin$ tables for all positions up to
`max_seq_len` once, then look them up in the forward pass.

**Apply RoPE to Q and K** (not V) before computing attention scores.

## 4.5 KV cache (inference only)

**What.** During generation, you produce one token at a time. Without
a cache, you would re-run the entire forward pass for each new token,
recomputing K and V for all previous positions. The cache stores K
and V as you go so you only compute them for the single newest token.

**Important.** The KV cache is **inference only**. If you try to use
it during training, you'll silently sever the autograd graph (the
cached tensors come from numpy slices that have no `_prev`). Assert
`not requires_grad` when entering the cache code path.

## 4.6 Checkpoints - verify Milestone 4

1. **Shape:** MHA applied to a $(B, T, d)$ input returns $(B, T, d)$.
2. **Causal mask:** Perturb token $T-1$; outputs at positions $0..T-2$
   must be unchanged.
3. **GQA:** With $h_q = 4, h_{kv} = 2$, output shape is still
   $(B, T, d)$ and the module has strictly fewer parameters than an
   MHA with the same $d$ and $h_q$.
4. **RoPE:** The function should be its own inverse - applying a
   forward rotation and then the transposed rotation should return
   the original tensor.

## 4.7 Common pitfalls

- **Forgetting the $\sqrt{d}$ scale.** Training will diverge or stall.
- **Off-by-one in the causal mask.** `triu(k=1)` is what you want:
  mask everything strictly above the diagonal. `triu(k=0)` would also
  mask the diagonal and prevent a token from looking at itself.
- **Applying RoPE to V.** Don't. Only Q and K.

---

# Milestone 5: Transformer block and model

**What and why.** Glue the pieces together. A **block** is one layer
of the Transformer: attention, then a feed-forward network, each with
its own residual connection and normalization. The **model** stacks
many blocks between an input embedding and an output head.

## 5.1 The feed-forward network (FFN)

**Standard form.** Two linear layers with a nonlinearity in between:

$$\text{FFN}(x) = W_2 \cdot \text{act}(W_1 x)$$

where $W_1$ maps $d \to 4d$ and $W_2$ maps $4d \to d$.

**SwiGLU form** (used by Llama, Mistral). Three linear layers:

$$\text{FFN}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$$

where $\odot$ is element-wise multiplication. $W_1$ is the **gate**,
$W_3$ is the **up projection**, $W_2$ is the **down projection**.

**Hidden dim for SwiGLU.** To keep the total parameter count
comparable to the standard $4d$ FFN, set $h = \lfloor 2/3 \cdot 4d \rfloor$,
rounded up to a multiple of some power of two (we use 8) for hardware
friendliness.

**Why SwiGLU?** The gating gives the FFN a data-dependent multiplicative
mask - similar capacity to bigger non-gated FFNs at fewer parameters.
Small but real improvement over GELU/ReLU.

## 5.2 The Transformer block (pre-norm)

```
function Block(x):
    x = x + attention(norm1(x))
    x = x + ffn      (norm2(x))
    return x
```

**Pre-norm vs. post-norm.** "Pre-norm" applies normalization *before*
each sub-layer and keeps the residual stream untouched between blocks.
This makes training much more stable and is what every modern LLM uses.

**Why residuals ($x = x + f(x)$)?** They provide a gradient highway: no
matter how deep the network, there is always an identity path from the
output back to the input. Without residuals, gradients through deep
stacks vanish.

## 5.3 The full model

```
function Transformer(ids):
    x = token_embedding(ids)                     # (B, T, d)
    x = dropout(x)
    for block in blocks:
        x = block(x, cos, sin)                   # passes RoPE tables through
    x = final_norm(x)
    logits = x @ head_weight + head_bias          # (B, T, vocab_size)
    return logits
```

## 5.4 Weight tying

**What.** Set the output projection weight equal to the input
embedding weight (transposed). I.e., if $E \in \mathbb{R}^{V \times d}$
is the embedding, the output head is $x E^\top$.

**Why.** Cuts $V \cdot d$ parameters (often a huge fraction of the
total for small models) and encourages a symmetric representation:
the same vector describes "what this token means" and "how similar my
output is to this token."

**Implementation gotcha.** If you create a new Linear layer and copy
the weight into it, your gradients will not flow back to the
embedding. Instead, in the forward pass, directly use the embedding
weight: `logits = x @ token_embedding.weight.transpose() + bias`. That
way the autograd graph connects the output loss directly to the
embedding parameter.

## 5.5 Checkpoints - verify Milestone 5

1. **Shape:** Model on $(B, T)$ integer IDs returns logits $(B, T, V)$.
2. **Variants:** Every combination of (RMSNorm/LayerNorm) × (SwiGLU/GELU/ReLU)
   × (tied/untied) should produce the right output shape.
3. **Tied < untied.** A weight-tied model has strictly fewer
   parameters than the untied version.
4. **Overfit a single batch.** Train on just one batch for ~200 steps.
   Loss should drop nearly to zero. If it doesn't, your autograd has a
   bug somewhere.

## 5.6 Common pitfalls

- **Post-norm.** Don't. Use pre-norm.
- **Forgetting the final norm.** There should be one `norm_f` between
  the last block and the output head.
- **Wrong SwiGLU wiring.** The gate is `SiLU(W_1 x)`, multiplied by
  `W_3 x`, then passed through $W_2$. A common mistake is to apply
  SiLU to the wrong projection.

---

# Milestone 6: Training

**What and why.** You've built the model. Now you need to teach it.
Training is: pick a batch, compute the loss, compute gradients,
update parameters, repeat. Do this millions of times.

## 6.1 Cross-entropy loss

**What.** For each position, the model outputs a vector of logits of
size $V$ (the vocab size). The loss is the negative log probability
the model assigned to the correct next token.

**Numerically stable form.** Use the log-sum-exp trick:

```
function cross_entropy(logits, targets):
    # logits shape: (B, T, V), targets shape: (B, T)
    max_val      = max(logits, axis=-1, keepdims=True)
    shifted      = logits - max_val                 # subtract max for stability
    log_sum_exp  = log(sum(exp(shifted), axis=-1, keepdims=True))
    log_probs    = shifted - log_sum_exp              # this is log softmax
    nll          = -log_probs[targets]                # pick out the target entries
    return mean(nll)
```

**Backward.** The gradient of cross-entropy w.r.t. logits has a
beautifully simple form:

$$\frac{\partial L}{\partial \text{logits}_i} = \frac{1}{N}(P_i - \mathbb{1}[i = \text{target}])$$

where $P$ is the softmax output. Compute it directly; do not
backprop through log-sum-exp.

## 6.2 Adam optimizer

**What.** A smarter version of gradient descent that tracks
per-parameter moving averages of the gradient ($m$) and its square
($v$), uses them for momentum + per-parameter adaptive step sizes,
and applies bias correction to undo the initialization bias.

**Algorithm.** For each parameter $\theta$ with gradient $g$ at step $t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$,
weight decay $= 0.01$.

**Weight decay.** Before the Adam update, shrink parameters slightly
toward zero: $\theta \leftarrow \theta - \eta \lambda \theta$. Acts as
a regularizer.

## 6.3 Learning rate schedule: warmup + cosine decay

**Why warmup?** At step 0 your weights are random and gradients are
noisy. Taking a big step immediately is dangerous. Linearly ramp the
learning rate from 0 to its target value over the first few hundred
steps.

**Why cosine decay?** After warmup, decrease the learning rate along a
cosine curve down to ~10% of the peak by the end of training. Empirically
works well and has no hyperparameters.

```
function lr(step):
    if step < warmup_steps:
        return lr_peak * step / warmup_steps
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + cos(pi * t))
```

## 6.4 Gradient clipping

**What.** Compute the L2 norm of *all* gradients concatenated. If it
exceeds a threshold (e.g., $1.0$), scale every gradient down by the
same factor so the total norm equals the threshold.

**Why.** Stops "gradient explosions" - occasional enormous gradients
from pathological examples that would otherwise destroy the weights.

## 6.5 Gradient accumulation

**What.** Want an effective batch size of $N$ but only fit $N/k$ in
memory? Do $k$ forward+backward passes accumulating gradients, then
take one optimizer step. Rescale gradients by $1/k$ to keep the
effective step size the same.

## 6.6 The training loop

```
function train(model, data, steps):
    opt = Adam(model.parameters(), lr=lr_peak, weight_decay=0.01)
    for step in range(steps):
        opt.lr = schedule(step)
        model.zero_grad()

        step_loss = 0
        for _ in range(grad_accum_steps):
            x, y = data.get_batch(batch_size)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            step_loss += loss

        if grad_accum_steps > 1:
            scale all parameter grads by 1/grad_accum_steps

        clip_grad_norm(model.parameters(), max_norm=1.0)
        opt.step()

        if step % log_interval == 0:
            log step_loss, current lr
```

## 6.7 Evaluation: perplexity

**Perplexity = $\exp(\text{mean cross-entropy})$.** It is the metric
LLM papers report. Intuition: a model with perplexity $P$ is "as
confused as if it had to choose uniformly between $P$ equally likely
options at every position." A perfectly random model over $V$ tokens
has perplexity $V$; a perfect model has perplexity $1$.

Always compute eval under `no_grad` and with `model.eval()`.

## 6.8 Checkpoints - verify Milestone 6

1. **Loss decreases.** On a simple repeating pattern (e.g., the
   sequence $[1,2,3,1,2,3,\ldots]$), training loss should drop
   dramatically within $\sim 100$ steps on a small model.
2. **Adam minimizes a quadratic.** Feed Adam the gradient of
   $(x-3)^2$ starting from $x=0$; after 200 steps $x$ should be near $3$.
3. **End-to-end.** After training on the repeating pattern, greedy
   generation from a prompt within the pattern should reproduce the
   pattern continuation.

## 6.9 Common pitfalls

- **Not zeroing gradients.** Your gradients accumulate forever and the
  optimizer takes wildly bigger steps each iteration. Call
  `zero_grad()` at the start of every step.
- **Forgetting `model.eval()` during validation.** Dropout stays on
  and your eval loss becomes noisy.
- **Training with the KV cache on.** It silently disables gradient
  flow through K and V. Assert it's off during training.

---

# Milestone 7: Text generation

**What and why.** Given a prompt, repeatedly: run the model, get
logits for the last position, sample a token from the resulting
distribution, append it, repeat.

## 7.1 Sampling strategies

- **Greedy** ($T \to 0$): pick the argmax. Deterministic, often boring
  and repetitive.
- **Temperature** $T$: divide logits by $T$ before softmax. $T < 1$
  sharpens the distribution (more focused); $T > 1$ flattens it (more
  random). $T \to 0$ is greedy; $T \to \infty$ is uniform.
- **Top-k**: zero out all but the $k$ most likely tokens, then sample.
- **Top-p (nucleus)**: sort tokens by probability, take the smallest
  set whose cumulative probability $\geq p$, renormalize, sample.
- **Repetition penalty**: for every token that has already appeared in
  the context, divide its logit by $\alpha$ if positive or multiply by
  $\alpha$ if negative (typical $\alpha = 1.1$). Discourages loops.

## 7.2 The generation loop

```
function generate(prompt_ids, max_new_tokens, ...):
    model.eval()
    cache = {}
    ids = list(prompt_ids)
    with no_grad():
        logits = model(ids, cache=cache)        # process the full prompt once
    for step in range(max_new_tokens):
        last_logits = logits[0, -1]              # (V,)
        apply repetition penalty
        scale by 1/temperature
        apply top-k / top-p
        probabilities = softmax(last_logits)
        next_token = sample from probabilities
        ids.append(next_token)
        with no_grad():
            logits = model([[next_token]], cache=cache)   # one-token forward
        yield next_token
```

## 7.3 Checkpoints - verify Milestone 7

1. **Determinism.** With `temperature=0` and a fixed seed, two runs
   from the same prompt produce identical output.
2. **KV-cache parity.** Generation with the KV cache must match the
   same generation without the cache (recomputing K,V every step) to
   within numerical tolerance.

---

# Milestone 8: Optimizations worth adding

Do these after everything above works end to end. In rough order of
educational value:

## 8.1 Gradient checkpointing

Trades compute for memory. Wrap each block in `checkpoint(fn)(x)`
which runs the forward under `no_grad` and reruns it during the
backward pass. Reduces peak memory by ~$N \times$ in an $N$-block
model at the cost of roughly 2x compute. See Milestone 1.6.

## 8.2 Save/load checkpoints with embedded config and tokenizer

Serialize the model weights, the `ModelConfig`, *and* the tokenizer
together. Then `Model.from_checkpoint(path)` can reconstruct the exact
architecture and tokenizer without you having to remember the
hyperparameters or locate a separate tokenizer file. Store the config
and tokenizer as JSON bytes in the same file as the weights.

The tokenizer is also cached separately on disk (keyed by data file,
tokenizer type, and vocab size) so that multiple training runs on the
same corpus can reuse it without retraining.

## 8.3 Optimizer state save/load

To resume training from a saved checkpoint, you also need to save
Adam's `m`, `v`, and step counter `t`. Save these as a sibling file
(e.g., `ckpt.npz` and `ckpt.npz.opt.npz`).

## 8.4 Mixed-precision-friendly computation

Keep parameter storage in low precision but promote to `float32` for
any operation where numerical stability matters: normalization,
softmax, loss calculation, optimizer math. This does not require
adding `bfloat16` - just be careful which dtype each op uses.

---

# Milestone 9: Testing strategy

Without tests you will accidentally break your autograd and not notice
until training mysteriously fails to converge. Minimum viable test
suite:

1. **Gradient checks** for every Tensor op using finite differences.
2. **Shape tests** for every module: feed random input, check output shape.
3. **Causal mask test**: perturb a future token, verify earlier outputs
   unchanged.
4. **Auto-parameter discovery test**: verify `Module.parameters()` finds
   all Tensor attributes, including nested in lists.
5. **Checkpoint roundtrip**: save a model (with tokenizer), load it,
   forward-pass output must match bit-for-bit and the tokenizer must
   encode/decode identically.
6. **Loss decreases** on a trivial structured dataset.
7. **End-to-end pattern learning**: train on a repeating sequence,
   assert the model can reproduce it with greedy decoding. This is
   the single most important test - it answers "does the whole thing
   work?"
8. **Gradient checkpointing equivalence**: loss and parameter grads
   with and without checkpointing must match.
9. **Tokenizer roundtrip**: `decode(encode(text)) == text` including
   empty strings and unicode.

---

# Milestone 10: Where does the time go?

A 5M-parameter model on CPU, batch 4, sequence 128:

| Component          | Share of forward time |
| :----------------- | :-------------------- |
| Embedding lookup   | $< 1\%$               |
| Attention          | $\sim 25\%$ per layer |
| FFN                | $\sim 18\%$ per layer |
| Everything else    | rounding error        |

Attention and FFN dominate. In real LLMs at scale, the FFN typically
wins because it has ~2-3x more parameters than attention per layer.
This is why the FFN hidden dim (and its efficient activation, SwiGLU)
matters so much for modern LLMs.

Run `scripts/profile.py` with `--n_embd`, `--n_head`, `--n_layer`, etc.
to see how the breakdown changes for different model sizes. It also
reports parameter counts, memory estimates, and throughput (tokens/sec).
Pass `--checkpoint path/to/model.npz` to profile a trained model
directly instead of constructing one from scratch.

---

# Milestone 11: Generation safety

When generating text, the model can only produce meaningful output for
positions it has positional embeddings for. RoPE frequencies are
precomputed for `block_size` positions, so `prompt_length +
max_new_tokens` must not exceed `block_size`. If it does, the RoPE
slice returns a wrong-shaped array and the model crashes or produces
garbage.

**Fix:** at the start of `generate()`, clamp `max_new_tokens` to
`block_size - len(prompt)`. If the prompt itself already fills the
block, raise a clear error rather than silently failing.

**Checkpoint:** call `generate()` with a prompt that leaves room for 5
tokens and request 1000. Verify it stops at 5 without crashing.

---

# Appendix A: Parameter count formula

For a Transformer with vocab $V$, model dim $d$, $L$ layers, $h$ heads,
FFN multiplier $m$ (so FFN hidden = $md$ for vanilla, or $\lfloor 2md/3 \rfloor$
for SwiGLU), tied weights, standard MHA:

- Embedding: $V \cdot d$
- Per block:
  - Attention: $4 d^2$ (Q, K, V, O projections - for standard MHA)
  - FFN (vanilla): $2 \cdot d \cdot md = 2 m d^2$
  - FFN (SwiGLU): $3 \cdot d \cdot \lfloor 2 m d / 3 \rfloor \approx 2 m d^2$
  - Norms: $2 d$ (tiny)
- Final norm + head bias: $d + V$ (head weight tied to embedding)

Total (ignoring bias terms): $Vd + L(4d^2 + 2md^2) + \ldots$. For
$V=32000$, $d=512$, $L=8$, $m=4$, you get $\sim 50\text{M}$ parameters,
half of which are the embedding.

---

# Appendix B: Recommended first experiment

Once it all compiles:

1. Download `shakespeare.txt` (~1 MB of Shakespeare plays).
2. Train a character-level model with $d=128$, 4 layers, 4 heads,
   block size 128, batch 16, 2000 steps, lr $10^{-3}$, cosine schedule.
3. On a laptop CPU, this finishes in minutes.
4. Generate from a prompt like `"ROMEO:"` with temperature 0.8.
5. You should see plausible-looking Shakespeare-ish text.

If the loss doesn't drop below ~2.0, something is wrong - most often
a broken gradient somewhere. Turn on gradient checks (Milestone 1.7)
and narrow it down.

---

# Appendix C: Glossary

- **Logit**: the raw model output before softmax. Larger = model
  thinks this token is more likely.
- **Token**: an integer ID produced by the tokenizer. Often a
  subword, sometimes a character, sometimes a whole word.
- **Context window / block size**: the maximum sequence length the
  model can handle in one forward pass.
- **Parameter**: a number that training adjusts. Every weight matrix
  entry is a parameter.
- **Gradient**: the derivative of the loss with respect to a
  parameter. Tells you how to change the parameter to decrease loss.
- **Activation**: the intermediate output of a layer during a forward
  pass (not a parameter - it's recomputed every forward).
- **Residual stream**: the sequence of intermediate outputs that
  flows through the transformer, updated by adding each block's
  output onto it.
- **Causal**: respecting time order - each position can only attend
  to positions at or before it.
- **Batch**: a group of independent sequences processed together for
  efficiency.

---

# Appendix D: Further reading

The original papers, in the order they're useful:

1. *Attention Is All You Need* (Vaswani et al., 2017) - the original
   Transformer. Ignore the encoder-decoder structure; you only need
   the decoder.
2. *Language Models are Unsupervised Multitask Learners* (GPT-2
   paper, Radford et al., 2019) - decoder-only formulation.
3. *RoFormer: Enhanced Transformer with Rotary Position Embedding*
   (Su et al., 2021) - RoPE.
4. *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019) -
   RMSNorm.
5. *GLU Variants Improve Transformer* (Shazeer, 2020) - SwiGLU and
   friends.
6. *LLaMA: Open and Efficient Foundation Language Models* (Touvron
   et al., 2023) - the recipe this project follows most closely.
7. *GQA: Training Generalized Multi-Query Transformer Models from
   Multi-Head Checkpoints* (Ainslie et al., 2023) - Grouped-Query
   Attention.

You now have a complete spec for a modern small LLM. Good luck.
