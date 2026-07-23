from __future__ import annotations

from typing import Any

import numpy as np


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "low"
    if probability < 0.60:
        return "medium"
    if probability < 0.80:
        return "high"
    return "critical"


def _prediction_label(prediction: int) -> str:
    return "attack" if int(prediction) == 1 else "benign"


def _build_previous_window_delta(
    previous_context: dict | None,
    prediction: int,
    attack_probability: float,
    feature_means: dict[str, float],
) -> dict[str, Any] | None:
    if previous_context is None:
        return None

    prev_signal = previous_context.get("model_signal", {})
    prev_summary = previous_context.get("window_summary", {})
    prev_means = prev_summary.get("feature_means", {})
    shared = [name for name in feature_means if name in prev_means]

    delta = {
        "attack_probability_delta": None,
        "prediction_changed": None,
        "top_feature_mean_deltas": [],
    }

    prev_prob = prev_signal.get("attack_probability")
    if isinstance(prev_prob, (int, float)):
        delta["attack_probability_delta"] = float(attack_probability) - float(prev_prob)

    prev_prediction = prev_signal.get("prediction")
    if isinstance(prev_prediction, str):
        delta["prediction_changed"] = prev_prediction != _prediction_label(prediction)
    elif isinstance(prev_prediction, (int, float)):
        delta["prediction_changed"] = int(prev_prediction) != int(prediction)

    ranked_shared = sorted(
        shared,
        key=lambda name: abs(float(feature_means[name]) - float(prev_means[name])),
        reverse=True,
    )
    for name in ranked_shared[:5]:
        delta["top_feature_mean_deltas"].append(
            {
                "name": name,
                "delta": float(feature_means[name]) - float(prev_means[name]),
                "current_mean": float(feature_means[name]),
                "previous_mean": float(prev_means[name]),
            }
        )

    return delta


def build_context_packet(
    window_id: int,
    X_window,
    feature_names: list[str],
    prediction: int,
    attack_probability: float,
    true_label: int | None = None,
    previous_context: dict | None = None,
    model_signal: dict | None = None,
    previous_model_signal: dict | None = None,
    stream_memory: dict | None = None,
    include_debug_notes: bool = False,
):
    """
    Build an LLM-ready context packet.

    The output should be structured, readable, and JSON serializable.
    The LLM is not called here.
    """
    X_window = np.asarray(X_window, dtype=float)
    if X_window.ndim != 2:
        raise ValueError("X_window must be a 2D array of shape [num_flows, feature_dim]")
    if len(feature_names) != X_window.shape[1]:
        raise ValueError("feature_names length must match window feature dimension")

    feature_means = {name: float(val) for name, val in zip(feature_names, X_window.mean(axis=0))}
    feature_max_values = {name: float(val) for name, val in zip(feature_names, X_window.max(axis=0))}
    feature_min_values = {name: float(val) for name, val in zip(feature_names, X_window.min(axis=0))}
    previous_window_delta = _build_previous_window_delta(
        previous_context=previous_context,
        prediction=prediction,
        attack_probability=attack_probability,
        feature_means=feature_means,
    )

    signal = model_signal or {
        "prediction": _prediction_label(prediction),
        "attack_probability": float(attack_probability),
        "risk_level": _risk_level(float(attack_probability)),
    }

    packet = {
        "window_id": int(window_id),
        "true_label": None if true_label is None else int(true_label),
        "model_signal": signal,
        "previous_model_signal": previous_model_signal,
        "stream_memory": stream_memory,
        "window_summary": {
            "num_flows": int(X_window.shape[0]),
            "num_features": int(X_window.shape[1]),
            "feature_means": feature_means,
            "feature_max_values": feature_max_values,
            "feature_min_values": feature_min_values,
            "previous_window_delta": previous_window_delta,
        },
    }
    if include_debug_notes:
        packet["debug_notes"] = [
            "This context packet was generated for future LLM assessment.",
            "The LLM consumes the structured signal and window summary, not raw training code.",
        ]
    if previous_context is not None:
        packet["window_summary"]["previous_window_delta"] = previous_window_delta
    return packet
