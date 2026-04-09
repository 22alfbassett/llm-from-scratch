"""Dataclass configs for model and training.

These are additive: existing kwargs-style APIs still work. New code should
prefer passing configs.
"""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.0
    tie_weights: bool = True
    act: str = "swiglu"
    norm: str = "rmsnorm"
    ffn_multiplier: int = 4  # FFN hidden dim factor (SwiGLU also applies 2/3)
    n_kv_head: Optional[int] = None  # for GQA; None -> same as n_head (standard MHA)
    grad_checkpoint: bool = False  # recompute block activations in backward

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
