import numpy as np

from src.llm.ollama_watcher import build_window_packet
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt


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


def test_blind_prompt_omits_detector_verdict_fields():
    X_window = np.random.randn(32, 4)
    feature_names = ["dur", "spkts", "dpkts", "rate"]
    packet = build_window_packet(X_window, feature_names, window_id=4)
    packet["model_signal"] = {
        "prediction_label": "attack",
        "attack_probability": 0.91,
        "risk_level": "critical",
    }
    prompt = build_user_prompt(packet, mode="blind")

    assert "prediction_label" not in prompt
    assert "attack_probability" not in prompt
    assert "risk_level" not in prompt
    assert "model_signal" not in prompt


def test_encoder_prompt_includes_representation_without_verdict_fields():
    X_window = np.random.randn(32, 4)
    feature_names = ["dur", "spkts", "dpkts", "rate"]
    packet = build_window_packet(X_window, feature_names, window_id=5)
    packet["model_signal"] = {
        "prediction_label": "attack",
        "attack_probability": 0.91,
        "risk_level": "critical",
        "embedding_summary": {
            "mean": 0.12,
            "std": 0.34,
            "min": -0.5,
            "max": 0.8,
            "l2_norm": 2.2,
        },
        "top_embedding_activations": [
            {"index": 7, "value": 0.8, "magnitude": 0.8},
        ],
    }
    packet["stream_memory"] = {
        "window_count": 3,
        "last_window_id": 4,
        "attack_probability_ema": 0.7,
        "embedding_l2_norm_trend": "rising",
        "rolling_embedding_l2_norm_mean": 2.0,
        "last_embedding_delta": {"embedding_l2_norm_delta": 0.4},
    }

    prompt = build_user_prompt(packet, mode="encoder")

    assert "mamba_sequence_representation" in prompt
    assert "embedding_summary" in prompt
    assert "top_embedding_activations" in prompt
    assert "embedding_l2_norm_trend" in prompt
    assert "prediction_label" not in prompt
    assert "attack_probability" not in prompt
    assert "risk_level" not in prompt


def test_system_prompt_describes_encoder_direction():
    assert "Mamba/S6 as a sequence encoder" in SYSTEM_PROMPT
    assert "not the final security verdict" in SYSTEM_PROMPT
    assert "learned numeric summary" in SYSTEM_PROMPT
    assert "stable facts and learned signal" in SYSTEM_PROMPT
    assert "Relationship hints" in SYSTEM_PROMPT
    assert "ground-truth answer" in SYSTEM_PROMPT
