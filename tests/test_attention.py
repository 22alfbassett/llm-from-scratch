from llm.attention import MultiHeadAttention
from llm.tensor import Tensor


def test_mha_output_shape():
    x = Tensor.randn(1, 4, 16)
    y = MultiHeadAttention(16, 4)(x)
    assert y.shape == (1, 4, 16)


def test_causal_mask_blocks_future():
    """Changing a future token must not affect the output at earlier positions."""
    import numpy as np

    from llm.tensor import np as dnp

    mha = MultiHeadAttention(8, 2)
    x1 = Tensor.randn(1, 4, 8)
    x2 = Tensor(dnp.array(x1.data))
    # perturb only the last token
    x2.data[0, 3, :] = x2.data[0, 3, :] + 10.0
    y1 = mha(x1).data
    y2 = mha(x2).data
    # positions 0..2 should be unchanged
    assert np.allclose(np.asarray(y1[0, :3]), np.asarray(y2[0, :3]), atol=1e-4)
