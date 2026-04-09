"""End-to-end: train a tiny model on a repeating pattern and verify that
greedy generation reproduces the pattern. This is the single test that
most honestly answers 'does this whole thing work?'."""

import math

from llm.config import ModelConfig
from llm.data import Dataset
from llm.tensor import set_seed
from llm.train import perplexity, train_loop
from llm.transformer import Transformer


def test_learns_repeating_pattern_and_generates_it():
    set_seed(0)

    # Repeating structured pattern the model must learn.
    pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tokens = pattern * 128
    dataset = Dataset(tokens, block_size=16)

    cfg = ModelConfig(
        vocab_size=12,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        dropout=0.0,
    )
    model = Transformer(cfg)

    train_loop(
        model,
        dataset,
        steps=120,
        lr=3e-3,
        batch_size=8,
        warmup_steps=10,
        grad_clip=1.0,
        log_interval=10_000,  # silence
    )

    # After training, perplexity should be dramatically below the
    # uniform-random baseline of vocab_size = 12.
    ppl = perplexity(model, dataset, iters=10, batch_size=8)
    assert ppl < 6.0, f"ppl={ppl:.2f} — model did not learn the pattern"

    # Greedy generation from a known prefix should continue the pattern.
    model.eval()
    prompt = [1, 2, 3, 4, 5]
    gen = list(model.generate(list(prompt), max_new_tokens=10, temperature=0))
    # After [1..5], the next tokens of the pattern are [6,7,8,9,10,1,2,3,4,5].
    expected = [6, 7, 8, 9, 10, 1, 2, 3, 4, 5]
    # Accept if at least 7/10 match — tiny model won't be perfect.
    matches = sum(1 for g, e in zip(gen, expected) if g == e)
    assert matches >= 7, f"generation {gen} did not reproduce pattern {expected}"


def test_perplexity_matches_exp_of_loss():
    """Perplexity must equal exp(estimate_loss) by definition."""
    from llm.train import estimate_loss

    set_seed(0)
    cfg = ModelConfig(vocab_size=10, n_embd=8, n_head=2, n_layer=1, block_size=4)
    model = Transformer(cfg)
    ds = Dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 20, block_size=4)

    set_seed(1)
    loss = estimate_loss(model, ds, iters=3, batch_size=2)
    set_seed(1)
    ppl = perplexity(model, ds, iters=3, batch_size=2)

    assert abs(ppl - math.exp(loss)) < 1e-5
