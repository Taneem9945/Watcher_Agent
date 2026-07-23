import json

from src.stream_memory import StreamMemory


def test_stream_memory_updates_and_serializes():
    memory = StreamMemory(history_limit=3, ema_alpha=0.5)

    first_signal = {
        "window_id": 1,
        "signal_rows": [
            {
                "window_id": 1,
                "row_index": 0,
                "prediction": 0,
                "prediction_label": "benign",
                "attack_probability": 0.2,
                "benign_probability": 0.8,
                "confidence": 0.8,
                "logit_margin": -1.0,
                "risk_level": "low",
            }
        ],
    }
    second_signal = {
        "window_id": 2,
        "signal_rows": [
            {
                "window_id": 2,
                "row_index": 0,
                "prediction": 1,
                "prediction_label": "attack",
                "attack_probability": 0.7,
                "benign_probability": 0.3,
                "confidence": 0.7,
                "logit_margin": 1.2,
                "risk_level": "high",
            }
        ],
    }
    third_signal = {
        "window_id": 3,
        "signal_rows": [
            {
                "window_id": 3,
                "row_index": 0,
                "prediction": 1,
                "prediction_label": "attack",
                "attack_probability": 0.9,
                "benign_probability": 0.1,
                "confidence": 0.95,
                "logit_margin": 2.1,
                "risk_level": "critical",
            }
        ],
    }

    snapshot1 = memory.update(first_signal)
    snapshot2 = memory.update(second_signal)
    snapshot3 = memory.update(third_signal)

    assert snapshot1["window_count"] == 1
    assert snapshot2["window_count"] == 2
    assert snapshot3["window_count"] == 3
    assert snapshot3["last_signal"]["prediction"] == 1
    assert snapshot3["attack_probability_trend"] == "rising"
    assert snapshot3["confidence_trend"] == "rising"
    assert snapshot3["logit_margin_trend"] == "rising"
    assert snapshot3["last_delta"]["prediction_changed"] is False
    assert snapshot3["attack_prediction_streak"] == 2
    assert snapshot3["suspicious_streak"] == 2
    assert snapshot3["anomaly_streak"] == 2
    assert abs(snapshot3["rolling_attack_probability_mean"] - ((0.2 + 0.7 + 0.9) / 3)) < 1e-9
    assert abs(snapshot3["rolling_confidence_mean"] - ((0.8 + 0.7 + 0.95) / 3)) < 1e-9
    assert abs(snapshot3["rolling_logit_margin_mean"] - ((-1.0 + 1.2 + 2.1) / 3)) < 1e-9

    json.dumps(snapshot3)
