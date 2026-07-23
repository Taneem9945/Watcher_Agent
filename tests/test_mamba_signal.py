import numpy as np
import torch

from src.mamba_model import MambaWatcher
from src.mamba_signal import build_mamba_signal_packet, infer_window_signal


def test_build_mamba_signal_packet_shape():
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    pooled = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)
    packet = build_mamba_signal_packet(
        window_id=5,
        logits=logits,
        pooled_embedding=pooled,
        true_label=1,
    )

    assert packet["window_id"] == 5
    assert packet["batch_size"] == 1
    assert packet["num_classes"] == 2
    assert len(packet["signal_rows"]) == 1
    row = packet["signal_rows"][0]
    assert row["prediction_label"] in {"attack", "benign"}
    assert "attack_probability" in row
    assert "logit_margin" in row
    assert "embedding_summary" in row
    assert "top_embedding_activations" in row


def test_infer_window_signal_runs():
    model = MambaWatcher(input_dim=4, d_model=8, d_state=4, d_conv=2, expand=2, num_classes=2)
    X_window = np.random.randn(12, 4).astype(np.float32)
    packet = infer_window_signal(model, X_window, window_id=2)

    assert packet["window_id"] == 2
    assert packet["batch_size"] == 1
    assert len(packet["signal_rows"]) == 1
