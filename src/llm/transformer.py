import json
from collections.abc import Iterator
from typing import Optional, Union

from .attention import MultiHeadAttention, precompute_rope_freqs
from .config import ModelConfig
from .nn import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    RMSNorm,
    Sequential,
    SiLU,
)
from .tensor import DTYPE, Tensor, checkpoint, no_grad, np


def _swiglu_hidden(n_embd: int, multiplier: int = 4, multiple_of: int = 8) -> int:
    """Hidden dim for SwiGLU, following LLaMA: round 2/3 * mult * d to multiple."""
    h = int(2 * multiplier * n_embd / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    def __init__(
        self,
        n_embd: int,
        dropout: float = 0.0,
        act: str = "swiglu",
        multiplier: int = 4,
    ):
        self.act_type = act
        if act == "swiglu":
            hidden = _swiglu_hidden(n_embd, multiplier=multiplier)
            self.w1 = Linear(n_embd, hidden)  # gate proj
            self.w3 = Linear(n_embd, hidden)  # up proj
            self.w2 = Linear(hidden, n_embd)  # down proj
            self.silu = SiLU()
        else:
            self.net = Sequential(
                [
                    Linear(n_embd, multiplier * n_embd),
                    GELU() if act == "gelu" else ReLU(),
                    Linear(multiplier * n_embd, n_embd),
                ]
            )
        self.dropout = Dropout(dropout)

    def __call__(self, x: Tensor) -> Tensor:
        if self.act_type == "swiglu":
            # SwiGLU: down( silu(gate(x)) * up(x) )
            return self.dropout(self.w2(self.silu(self.w1(x)) * self.w3(x)))
        return self.dropout(self.net(x))


class Block(Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        act: str = "swiglu",
        norm: str = "rmsnorm",
        ffn_multiplier: int = 4,
        n_kv_head: Optional[int] = None,
    ):
        self.sa = MultiHeadAttention(n_embd, n_head, dropout, n_kv_head=n_kv_head)
        self.ffwd = FeedForward(n_embd, dropout, act=act, multiplier=ffn_multiplier)
        Norm = RMSNorm if norm == "rmsnorm" else LayerNorm
        self.norm1, self.norm2 = Norm(n_embd), Norm(n_embd)

    def __call__(
        self,
        x: Tensor,
        cos: np.ndarray = None,
        sin: np.ndarray = None,
        cache: dict = None,
        layer_idx: int = None,
    ) -> Tensor:
        x = x + self.sa(self.norm1(x), cos=cos, sin=sin, cache=cache, layer_idx=layer_idx)
        x = x + self.ffwd(self.norm2(x))
        return x


class Transformer(Module):
    """Decoder-only transformer with RoPE.

    Can be built either from a ``ModelConfig`` or from positional args (legacy).
    Checkpoints now embed the config so ``Transformer.from_checkpoint(path)``
    reconstructs the correct architecture automatically.
    """

    def __init__(
        self,
        vocab_size: Union[int, ModelConfig],
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 128,
        dropout: float = 0.0,
        tie_weights: bool = True,
        act: str = "swiglu",
        norm: str = "rmsnorm",
    ):
        if isinstance(vocab_size, ModelConfig):
            cfg = vocab_size
        else:
            cfg = ModelConfig(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                block_size=block_size,
                dropout=dropout,
                tie_weights=tie_weights,
                act=act,
                norm=norm,
            )
        self.config = cfg
        self.block_size = cfg.block_size
        self.tie_weights = cfg.tie_weights

        self.token_embedding = Embedding(cfg.vocab_size, cfg.n_embd)
        self.emb_dropout = Dropout(cfg.dropout)
        self.blocks = [
            Block(
                cfg.n_embd,
                cfg.n_head,
                cfg.dropout,
                act=cfg.act,
                norm=cfg.norm,
                ffn_multiplier=cfg.ffn_multiplier,
                n_kv_head=cfg.n_kv_head,
            )
            for _ in range(cfg.n_layer)
        ]
        Norm = RMSNorm if cfg.norm == "rmsnorm" else LayerNorm
        self.norm_f = Norm(cfg.n_embd)

        # Output head.
        # Tied: reuse token_embedding.weight directly (no new param), add a bias.
        # Untied: independent Linear.
        if cfg.tie_weights:
            self.lm_head_bias = Tensor(np.zeros((1, cfg.vocab_size)))
            self.lm_head = None
        else:
            self.lm_head = Linear(cfg.n_embd, cfg.vocab_size)
            self.lm_head_bias = None

        self.cos, self.sin = precompute_rope_freqs(cfg.n_embd // cfg.n_head, cfg.block_size)

    def __call__(self, idx: Union[list, np.ndarray], cache: dict = None) -> Tensor:
        x = self.emb_dropout(self.token_embedding(np.array(idx)))
        use_ckpt = (
            self.config.grad_checkpoint and self.training and Tensor.requires_grad and cache is None
        )
        for i, block in enumerate(self.blocks):
            if use_ckpt:
                cos_, sin_, idx_ = self.cos, self.sin, i
                blk = block
                x = checkpoint(
                    lambda inp, blk=blk, cos_=cos_, sin_=sin_, idx_=idx_: blk(
                        inp, cos=cos_, sin=sin_, cache=None, layer_idx=idx_
                    ),
                    x,
                )
            else:
                x = block(x, cos=self.cos, sin=self.sin, cache=cache, layer_idx=i)
        x = self.norm_f(x)
        if self.tie_weights:
            # logits = x @ W_emb.T + bias
            return x.matmul(self.token_embedding.weight.transpose(0, 1)) + self.lm_head_bias
        return self.lm_head(x)

    def parameters(self) -> Iterator[Tensor]:
        seen = set()
        mods = [self.token_embedding, self.emb_dropout, *self.blocks, self.norm_f]
        if self.lm_head is not None:
            mods.append(self.lm_head)
        for mod in mods:
            for p in mod.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p
        if self.lm_head_bias is not None and id(self.lm_head_bias) not in seen:
            seen.add(id(self.lm_head_bias))
            yield self.lm_head_bias

    # ------------------------------------------------------------------
    # Checkpointing with embedded config
    # ------------------------------------------------------------------
    def save(self, path: str, tokenizer=None) -> None:
        params = {f"arr_{i}": p.data for i, p in enumerate(self.parameters())}
        cfg_bytes = json.dumps(self.config.to_dict()).encode("utf-8")
        params["_config_"] = np.frombuffer(cfg_bytes, dtype=np.uint8)
        if tokenizer is not None:
            tok_bytes = json.dumps(tokenizer.to_dict()).encode("utf-8")
            params["_tokenizer_"] = np.frombuffer(tok_bytes, dtype=np.uint8)
        np.savez(path, **params)

    def load(self, path: str) -> None:
        import numpy as _np

        with _np.load(path) as data:
            params = list(self.parameters())
            for i, p in enumerate(params):
                key = f"arr_{i}"
                if key not in data.files:
                    raise ValueError(f"checkpoint missing {key}")
                arr = data[key]
                if p.data.shape != arr.shape:
                    raise ValueError(
                        f"shape mismatch for {key}: ckpt {arr.shape} vs model {p.data.shape}"
                    )
                p.data = np.array(arr, dtype=DTYPE)

    @classmethod
    def from_checkpoint(cls, path: str) -> tuple["Transformer", object]:
        """Load a model (and its tokenizer) from a checkpoint.

        Returns ``(model, tokenizer)``.  The tokenizer is ``None`` only
        if the checkpoint was saved without one.
        """
        import numpy as _np

        from .data import tokenizer_from_dict

        with _np.load(path) as data:
            if "_config_" not in data.files:
                raise ValueError(f"{path} has no embedded config; cannot auto-load")
            cfg_bytes = bytes(data["_config_"].tolist())
            cfg = ModelConfig.from_dict(json.loads(cfg_bytes.decode("utf-8")))
            tokenizer = None
            if "_tokenizer_" in data.files:
                tok_bytes = bytes(data["_tokenizer_"].tolist())
                tokenizer = tokenizer_from_dict(json.loads(tok_bytes.decode("utf-8")))
        model = cls(cfg)
        model.load(path)
        return model, tokenizer

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_next(
        last_logits,
        idx,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        """Pick one next token from a logit vector, applying penalty and sampling."""
        import numpy as snp

        if repetition_penalty != 1.0:
            for token in set(idx):
                if last_logits[token] > 0:
                    last_logits[token] /= repetition_penalty
                else:
                    last_logits[token] *= repetition_penalty

        if temperature <= 0:
            return int(last_logits.argmax())

        scaled = last_logits / temperature
        if top_k is not None:
            scaled[scaled < np.partition(scaled, -top_k)[-top_k]] = -float("Inf")
        exps = np.exp(scaled - np.max(scaled))
        probs = exps / np.sum(exps)

        if top_p is not None:
            order = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[order])
            drop = cum > top_p
            drop[1:] = drop[:-1].copy()
            drop[0] = False
            probs[order[drop]] = 0
            probs /= np.sum(probs)

        probs_cpu = probs.get() if hasattr(probs, "get") else probs
        return int(snp.random.choice(len(probs_cpu), p=probs_cpu))

    def generate(
        self,
        idx: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        verbose: bool = False,
    ):

        self.eval()
        cache = {}
        prompt_len = len(idx)
        remaining = self.block_size - prompt_len
        if remaining <= 0:
            raise ValueError(
                f"Prompt length ({prompt_len}) already meets or exceeds "
                f"block_size ({self.block_size}). Cannot generate new tokens."
            )
        max_new_tokens = min(max_new_tokens, remaining)
        idx_cond = [idx[-self.block_size :]]
        with no_grad():
            logits = self(idx_cond, cache=cache)

        for i in range(max_new_tokens):
            if verbose:
                print(f"\rtoken: {i + 1}/{max_new_tokens}", end="", flush=True)
            next_token = self._sample_next(
                logits.data[0][-1].copy(),
                idx,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )

            idx.append(int(next_token))
            yield int(next_token)

            with no_grad():
                logits = self([[next_token]], cache=cache)
        if verbose:
            print()
