import numpy as np
import torch

from src.mamba_model import MAMBA_AVAILABLE, MAMBA_SOURCE, MambaWatcher


def test_model_forward_shape():
    assert MAMBA_AVAILABLE
    model = MambaWatcher(input_dim=40)
    x = torch.tensor(np.random.randn(2, 100, 40), dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (2, 2)
    assert MAMBA_SOURCE == "official-vendored"


def test_model_encode_accepts_padding_mask():
    model = MambaWatcher(input_dim=4, d_model=8, d_state=4, d_conv=2, expand=2, pooling="mean")
    x = torch.tensor(np.random.randn(2, 5, 4), dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)

    embedding = model.encode(x, mask=mask)
    logits = model(x, mask=mask)

    assert embedding.shape == (2, 8)
    assert logits.shape == (2, 2)
