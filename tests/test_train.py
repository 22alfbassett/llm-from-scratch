"""Smoke test: loss should decrease when overfitting a tiny dataset."""

from llm.config import ModelConfig
from llm.data import Dataset
from llm.train import train_loop
from llm.transformer import Transformer


def test_loss_decreases_on_tiny_corpus():
    # structured repeating pattern so the model has something learnable
    pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tokens = pattern * 64
    ds = Dataset(tokens, block_size=8)

    cfg = ModelConfig(
        vocab_size=12,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=8,
        dropout=0.0,
    )
    model = Transformer(cfg)

    history = train_loop(model, ds, steps=80, lr=3e-3, batch_size=8, warmup_steps=5, grad_clip=1.0)
    losses = history["train"]
    assert len(losses) == 80
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial - 0.3, f"loss did not decrease: {initial} -> {final}"
