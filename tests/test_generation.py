import numpy as np

from llm.config import ModelConfig
from llm.tensor import set_seed
from llm.transformer import Transformer


def _small_model(vocab=20):
    return Transformer(
        ModelConfig(
            vocab_size=vocab,
            n_embd=16,
            n_head=2,
            n_layer=2,
            block_size=16,
            dropout=0.0,
        )
    )


def test_kv_cache_parity():
    """Generation with KV-cache must match recomputing from scratch (greedy)."""
    set_seed(0)
    model = _small_model()
    model.eval()
    prompt = [1, 2, 3, 4]

    set_seed(123)
    cached = list(model.generate(list(prompt), max_new_tokens=6, temperature=0))

    # Recompute without cache by running forward per step via a fresh list.
    from llm.tensor import no_grad

    uncached = []
    ctx = list(prompt)
    with no_grad():
        for _ in range(6):
            logits = model([ctx[-model.block_size :]])
            last = logits.data[0][-1]
            nxt = int(np.asarray(last.get() if hasattr(last, "get") else last).argmax())
            uncached.append(nxt)
            ctx.append(nxt)

    assert cached == uncached


def test_deterministic_generation_seeded():
    set_seed(0)
    model = _small_model()
    model.eval()
    a = list(model.generate([1, 2, 3], max_new_tokens=5, temperature=0))
    b = list(model.generate([1, 2, 3], max_new_tokens=5, temperature=0))
    assert a == b
