from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ollama_client import OllamaClient, OllamaError
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise OllamaError(f"Ollama did not return valid JSON: {text[:400]}") from exc


def _feature_family(name: str) -> str:
    if name.startswith("proto_"):
        return "protocol"
    if name.startswith("service_"):
        return "service"
    if name.startswith("state_"):
        return "state"
    return "numeric"


def _build_salient_features(
    feature_means: dict[str, float],
    feature_max_values: dict[str, float],
    feature_min_values: dict[str, float],
    top_k: int = 12,
) -> dict[str, list[dict[str, float | str]]]:
    ranked = sorted(feature_means.items(), key=lambda item: abs(item[1]), reverse=True)
    salient = []
    for name, mean in ranked[:top_k]:
        salient.append(
            {
                "name": name,
                "family": _feature_family(name),
                "mean": float(mean),
                "min": float(feature_min_values[name]),
                "max": float(feature_max_values[name]),
                "direction": "elevated" if mean > 0 else "suppressed",
            }
        )

    positive = [item for item in salient if item["mean"] > 0]
    negative = [item for item in salient if item["mean"] <= 0]
    return {
        "top_absolute": salient,
        "top_positive": positive,
        "top_negative": negative,
    }


def _build_family_snapshot(salient_features: dict[str, list[dict[str, float | str]]]) -> dict[str, list[dict[str, float | str]]]:
    grouped = {"numeric": [], "protocol": [], "service": [], "state": []}
    for item in salient_features["top_absolute"]:
        grouped[item["family"]].append(item)
    return grouped


def _build_security_signals(family_snapshot: dict[str, list[dict[str, float | str]]]) -> dict[str, Any]:
    signals: dict[str, Any] = {}

    top_service = family_snapshot["service"][:3]
    top_state = family_snapshot["state"][:3]
    top_numeric = family_snapshot["numeric"][:4]
    top_protocol = family_snapshot["protocol"][:3]

    if top_service:
        signals["service_signals"] = [item["name"] for item in top_service]
    if top_protocol:
        signals["protocol_signals"] = [item["name"] for item in top_protocol]
    if top_state:
        signals["state_signals"] = [item["name"] for item in top_state]
    if top_numeric:
        signals["numeric_signals"] = [
            {
                "name": item["name"],
                "mean": item["mean"],
                "trend": item["direction"],
            }
            for item in top_numeric
        ]

    heuristic_notes: list[str] = []
    if any(item["name"] == "service_http" for item in family_snapshot["service"]):
        heuristic_notes.append("HTTP-related activity is elevated in this window.")
    if any(item["name"] == "service_ssh" for item in family_snapshot["service"]):
        heuristic_notes.append("SSH-related activity is elevated in this window.")
    if any(item["name"] == "proto_tcp" for item in family_snapshot["protocol"]):
        heuristic_notes.append("TCP transport appears more active than baseline.")
    if any(item["name"] == "proto_udp" for item in family_snapshot["protocol"]):
        heuristic_notes.append("UDP transport appears more active than baseline.")
    if any(item["name"] == "state_CON" and item["mean"] > 0 for item in family_snapshot["state"]):
        heuristic_notes.append("Connected flows are above the local window baseline.")
    if any(item["name"] == "state_INT" and item["mean"] < 0 for item in family_snapshot["state"]):
        heuristic_notes.append("Interrupted flows are below the local window baseline.")
    if any(item["name"] == "sttl" for item in family_snapshot["numeric"]):
        heuristic_notes.append("Source TTL is a strong outlier relative to the window baseline.")
    if any(item["name"] == "ct_state_ttl" for item in family_snapshot["numeric"]):
        heuristic_notes.append("Connection state and TTL behavior is unusual in this window.")

    signals["heuristic_notes"] = heuristic_notes
    return signals


def build_window_packet(
    X_window,
    feature_names: list[str],
    window_id: int,
    previous_packet: dict | None = None,
    include_debug_notes: bool = False,
):
    X_window = np.asarray(X_window, dtype=float)
    if X_window.ndim != 2:
        raise ValueError("X_window must be 2D: [window_size, feature_dim]")
    if len(feature_names) != X_window.shape[1]:
        raise ValueError("feature_names length must match feature dimension")

    feature_means = {name: float(val) for name, val in zip(feature_names, X_window.mean(axis=0))}
    feature_max_values = {name: float(val) for name, val in zip(feature_names, X_window.max(axis=0))}
    feature_min_values = {name: float(val) for name, val in zip(feature_names, X_window.min(axis=0))}
    salient_features = _build_salient_features(feature_means, feature_max_values, feature_min_values)
    family_snapshot = _build_family_snapshot(salient_features)
    security_signals = _build_security_signals(family_snapshot)
    previous_window_delta = None
    if previous_packet is not None:
        prev_summary = previous_packet.get("window_summary", {})
        prev_means = prev_summary.get("feature_means", {})
        shared = [name for name in feature_means if name in prev_means]
        ranked_shared = sorted(
            shared,
            key=lambda name: abs(float(feature_means[name]) - float(prev_means[name])),
            reverse=True,
        )
        previous_window_delta = {
            "top_feature_mean_deltas": [
                {
                    "name": name,
                    "delta": float(feature_means[name]) - float(prev_means[name]),
                    "current_mean": float(feature_means[name]),
                    "previous_mean": float(prev_means[name]),
                }
                for name in ranked_shared[:5]
            ]
        }
        prev_signals = prev_summary.get("security_signals", {})
        prev_state = prev_signals.get("state_signals", [])
        curr_state = security_signals.get("state_signals", [])
        previous_window_delta["state_signal_changed"] = prev_state != curr_state

    return {
        "window_id": int(window_id),
        "window_summary": {
            "num_flows": int(X_window.shape[0]),
            "num_features": int(X_window.shape[1]),
            "salient_features": salient_features,
            "family_snapshot": family_snapshot,
            "security_signals": security_signals,
            "previous_window_delta": previous_window_delta,
            "feature_means": feature_means,
            "feature_max_values": feature_max_values,
            "feature_min_values": feature_min_values,
        },
    }


@dataclass
class OllamaWatcher:
    client: OllamaClient

    def assess_context_packet(
        self,
        context_packet: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context_packet)},
        ]
        response_text = self.client.chat_raw(messages=messages, stream=False)
        assessment = _safe_json_loads(response_text)
        assessment["window_id"] = int(context_packet.get("window_id", -1))
        assessment["llm_input_packet"] = context_packet
        assessment["raw_response_text"] = response_text
        return assessment

    def assess_window(
        self,
        X_window,
        feature_names: list[str],
        window_id: int,
        previous_packet: dict | None = None,
    ) -> dict[str, Any]:
        packet = build_window_packet(
            X_window=X_window,
            feature_names=feature_names,
            window_id=window_id,
            previous_packet=previous_packet,
        )
        return self.assess_context_packet(packet)
