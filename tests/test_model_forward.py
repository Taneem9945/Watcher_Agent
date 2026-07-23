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
