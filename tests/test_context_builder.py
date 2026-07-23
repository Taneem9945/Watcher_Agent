import json

import numpy as np

from src.context_builder import build_context_packet


def test_context_builder_serializable():
    X_window = np.random.randn(100, 4)
    feature_names = ["duration", "bytes", "packets", "flow_rate"]
    packet = build_context_packet(
        window_id=7,
        X_window=X_window,
        feature_names=feature_names,
        prediction=1,
        attack_probability=0.83,
        true_label=0,
    )

    json.dumps(packet)
    assert "model_signal" in packet
    assert "previous_model_signal" in packet
    assert "window_summary" in packet
    assert "debug_notes" not in packet
    assert "previous_window_delta" in packet["window_summary"]
    assert packet["window_summary"]["previous_window_delta"] is None


def test_context_builder_debug_notes_optional():
    X_window = np.random.randn(8, 4)
    feature_names = ["duration", "bytes", "packets", "flow_rate"]
    packet = build_context_packet(
        window_id=8,
        X_window=X_window,
        feature_names=feature_names,
        prediction=0,
        attack_probability=0.12,
        include_debug_notes=True,
    )

    assert "debug_notes" in packet


def test_context_builder_previous_signal():
    X_window = np.random.randn(8, 4)
    feature_names = ["duration", "bytes", "packets", "flow_rate"]
    packet = build_context_packet(
        window_id=9,
        X_window=X_window,
        feature_names=feature_names,
        prediction=1,
        attack_probability=0.72,
        previous_model_signal={"prediction_label": "benign", "attack_probability": 0.15},
    )

    assert packet["previous_model_signal"] is not None


def test_context_builder_stream_memory():
    X_window = np.random.randn(8, 4)
    feature_names = ["duration", "bytes", "packets", "flow_rate"]
    packet = build_context_packet(
        window_id=10,
        X_window=X_window,
        feature_names=feature_names,
        prediction=0,
        attack_probability=0.11,
        stream_memory={
            "window_count": 2,
            "attack_probability_ema": 0.3,
            "attack_probability_trend": "rising",
        },
    )

    assert packet["stream_memory"]["window_count"] == 2
