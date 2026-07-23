import numpy as np

from src.llm.ollama_watcher import build_window_packet


def test_build_window_packet_shape():
    X_window = np.random.randn(32, 4)
    feature_names = ["dur", "spkts", "dpkts", "rate"]
    packet = build_window_packet(X_window, feature_names, window_id=3)
    assert packet["window_id"] == 3
    assert packet["window_summary"]["num_flows"] == 32
    assert packet["window_summary"]["num_features"] == 4
    assert "feature_means" in packet["window_summary"]
    assert "salient_features" in packet["window_summary"]
    assert "family_snapshot" in packet["window_summary"]
    assert "top_absolute" in packet["window_summary"]["salient_features"]
    assert "previous_window_delta" in packet["window_summary"]
    assert "temporal_notes" not in packet
