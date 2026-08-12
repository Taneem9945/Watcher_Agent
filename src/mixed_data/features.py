from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .windowing import EventWindow


BASE_FEATURES = (
    "offset_seconds",
    "src_port",
    "dst_port",
    "host_present",
    "src_ip_present",
    "dst_ip_present",
    "username_present",
    "uid_present",
    "missing_field_count",
)


@dataclass(frozen=True)
class WindowTensorBatch:
    X: np.ndarray
    mask: np.ndarray
    feature_names: list[str]
    metadata: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.X.shape),
            "mask_shape": list(self.mask.shape),
            "feature_count": len(self.feature_names),
            "window_count": len(self.metadata),
            "feature_names": self.feature_names,
            "metadata_preview": self.metadata[:3],
        }

    def metadata_dicts(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.metadata]


def _all_event_dicts(windows: Sequence[EventWindow]) -> list[dict[str, Any]]:
    return [event for window in windows for event in window.events]


def _feature_keys(windows: Sequence[EventWindow]) -> list[str]:
    keys = set()
    for event in _all_event_dicts(windows):
        keys.update((event.get("features") or {}).keys())
    return sorted(keys)


def _categorical_values(windows: Sequence[EventWindow], key: str) -> list[str]:
    values = {str(event.get(key)) for event in _all_event_dicts(windows) if event.get(key) not in (None, "")}
    return sorted(values)


def build_feature_names(windows: Sequence[EventWindow]) -> list[str]:
    source_features = [f"source_type={value}" for value in _categorical_values(windows, "source_type")]
    event_type_features = [f"event_type={value}" for value in _categorical_values(windows, "event_type")]
    parser_features = [f"parser={value}" for value in _categorical_values(windows, "parser")]
    role_features = [f"role={value}" for value in _categorical_values(windows, "role")]
    return [
        *BASE_FEATURES,
        *source_features,
        *event_type_features,
        *parser_features,
        *role_features,
        *[f"feature.{key}" for key in _feature_keys(windows)],
    ]


def _numeric_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def vectorize_event(event: dict[str, Any], window: EventWindow, feature_names: Sequence[str]) -> np.ndarray:
    values = {
        "offset_seconds": max(0.0, float(event["timestamp_epoch"]) - float(window.start_epoch)),
        "src_port": _numeric_or_zero(event.get("src_port")),
        "dst_port": _numeric_or_zero(event.get("dst_port")),
        "host_present": float(event.get("host") not in (None, "")),
        "src_ip_present": float(event.get("src_ip") not in (None, "")),
        "dst_ip_present": float(event.get("dst_ip") not in (None, "")),
        "username_present": float(event.get("username") not in (None, "")),
        "uid_present": float(event.get("uid") not in (None, "")),
        "missing_field_count": float(len(event.get("missing_fields") or [])),
    }
    features = event.get("features") or {}
    vector = []
    for name in feature_names:
        if name in values:
            vector.append(values[name])
        elif name.startswith("source_type="):
            vector.append(float(event.get("source_type") == name.split("=", 1)[1]))
        elif name.startswith("event_type="):
            vector.append(float(event.get("event_type") == name.split("=", 1)[1]))
        elif name.startswith("parser="):
            vector.append(float(event.get("parser") == name.split("=", 1)[1]))
        elif name.startswith("role="):
            vector.append(float(event.get("role") == name.split("=", 1)[1]))
        elif name.startswith("feature."):
            vector.append(_numeric_or_zero(features.get(name.removeprefix("feature."))))
        else:
            vector.append(0.0)
    return np.asarray(vector, dtype=np.float32)


def _standardize_real_events(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    real_rows = X[mask.astype(bool)]
    if real_rows.size == 0:
        return X
    means = real_rows.mean(axis=0)
    stds = real_rows.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    scaled = X.copy()
    scaled[mask.astype(bool)] = (real_rows - means) / stds
    return scaled.astype(np.float32)


def windows_to_tensor_batch(
    windows: Sequence[EventWindow],
    max_events: int | None = None,
    standardize: bool = True,
) -> WindowTensorBatch:
    if not windows:
        return WindowTensorBatch(
            X=np.empty((0, 0, 0), dtype=np.float32),
            mask=np.empty((0, 0), dtype=np.float32),
            feature_names=[],
            metadata=[],
        )

    feature_names = build_feature_names(windows)
    sequence_len = max_events if max_events is not None else max(window.event_count for window in windows)
    if sequence_len <= 0:
        raise ValueError("max_events must be positive")

    X = np.zeros((len(windows), sequence_len, len(feature_names)), dtype=np.float32)
    mask = np.zeros((len(windows), sequence_len), dtype=np.float32)
    metadata = []
    for window_index, window in enumerate(windows):
        selected_events = window.events[:sequence_len]
        for event_index, event in enumerate(selected_events):
            X[window_index, event_index] = vectorize_event(event, window, feature_names)
            mask[window_index, event_index] = 1.0
        metadata.append(
            {
                "window_id": window.window_id,
                "time_bucket_index": window.time_bucket_index,
                "start_timestamp": window.start_timestamp,
                "end_timestamp": window.end_timestamp,
                "event_count": window.event_count,
                "encoded_event_count": len(selected_events),
                "truncated_event_count": max(0, window.event_count - len(selected_events)),
                "source_counts": dict(window.source_counts),
                "event_type_counts": dict(window.event_type_counts),
            }
        )

    if standardize:
        X = _standardize_real_events(X, mask)
    return WindowTensorBatch(X=X, mask=mask, feature_names=feature_names, metadata=metadata)


def window_metadata_to_dicts(windows: Sequence[EventWindow]) -> list[dict[str, Any]]:
    return [asdict(window) for window in windows]
