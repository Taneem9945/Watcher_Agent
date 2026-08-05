from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _first_signal_row(signal_packet: dict[str, Any]) -> dict[str, Any]:
    rows = signal_packet.get("signal_rows", [])
    if not rows:
        raise ValueError("signal_packet must contain at least one signal row")
    row = rows[0]
    if not isinstance(row, dict):
        raise ValueError("signal row must be a dictionary")
    return row


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient"
    if values[-1] > values[0] + 1e-6:
        return "rising"
    if values[-1] < values[0] - 1e-6:
        return "falling"
    return "stable"


def _rolling_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@dataclass
class StreamMemory:
    """Lightweight state for the watcher pipeline.

    This is app-layer memory, not a true recurrent Mamba state. It keeps recent
    signal summaries so the context builder can describe change across windows.
    """

    history_limit: int = 5
    ema_alpha: float = 0.35
    anomaly_probability_threshold: float = 0.60
    window_count: int = 0
    last_window_id: int | None = None
    last_signal_row: dict[str, Any] | None = None
    last_context_packet: dict[str, Any] | None = None
    attack_probability_ema: float | None = None
    confidence_ema: float | None = None
    logit_margin_ema: float | None = None
    prediction_streak: int = 0
    suspicious_streak: int = 0
    anomaly_streak: int = 0
    attack_prediction_streak: int = 0
    last_prediction: int | None = None
    recent_signals: list[dict[str, Any]] = field(default_factory=list)
    recent_deltas: list[dict[str, Any]] = field(default_factory=list)
    attack_probability_history: list[float] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    logit_margin_history: list[float] = field(default_factory=list)
    risk_level_history: list[str] = field(default_factory=list)
    embedding_l2_norm_history: list[float] = field(default_factory=list)

    def update(
        self,
        signal_packet: dict[str, Any],
        context_packet: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = _first_signal_row(signal_packet)
        current_probability = _safe_float(row.get("attack_probability"))
        current_confidence = _safe_float(row.get("confidence"))
        current_margin = _safe_float(row.get("logit_margin"))
        current_embedding_summary = row.get("embedding_summary") if isinstance(row.get("embedding_summary"), dict) else {}
        current_embedding_l2_norm = _safe_float(current_embedding_summary.get("l2_norm"))
        current_prediction = row.get("prediction")
        current_risk_level = row.get("risk_level")
        current_window_id = signal_packet.get("window_id")
        current_is_attack_prediction = isinstance(current_prediction, (int, float)) and int(current_prediction) == 1
        current_is_suspicious = isinstance(current_risk_level, str) and current_risk_level in {"medium", "high", "critical"}
        current_is_anomaly = (
            (current_probability is not None and current_probability >= self.anomaly_probability_threshold)
            or current_is_suspicious
            or current_is_attack_prediction
        )

        delta = self._build_delta(row)

        self.window_count += 1
        self.last_window_id = int(current_window_id) if isinstance(current_window_id, (int, float)) else None
        self.last_context_packet = context_packet
        self.last_signal_row = row
        self.last_prediction = int(current_prediction) if isinstance(current_prediction, (int, float)) else None

        if current_probability is not None:
            self.attack_probability_history.append(current_probability)
            if len(self.attack_probability_history) > self.history_limit:
                self.attack_probability_history = self.attack_probability_history[-self.history_limit :]
            if self.attack_probability_ema is None:
                self.attack_probability_ema = current_probability
            else:
                self.attack_probability_ema = (
                    self.ema_alpha * current_probability
                    + (1.0 - self.ema_alpha) * self.attack_probability_ema
                )

        if current_confidence is not None:
            self.confidence_history.append(current_confidence)
            if len(self.confidence_history) > self.history_limit:
                self.confidence_history = self.confidence_history[-self.history_limit :]
            if self.confidence_ema is None:
                self.confidence_ema = current_confidence
            else:
                self.confidence_ema = self.ema_alpha * current_confidence + (1.0 - self.ema_alpha) * self.confidence_ema

        if current_margin is not None:
            self.logit_margin_history.append(current_margin)
            if len(self.logit_margin_history) > self.history_limit:
                self.logit_margin_history = self.logit_margin_history[-self.history_limit :]
            if self.logit_margin_ema is None:
                self.logit_margin_ema = current_margin
            else:
                self.logit_margin_ema = self.ema_alpha * current_margin + (1.0 - self.ema_alpha) * self.logit_margin_ema

        if isinstance(current_risk_level, str):
            self.risk_level_history.append(current_risk_level)
            if len(self.risk_level_history) > self.history_limit:
                self.risk_level_history = self.risk_level_history[-self.history_limit :]

        if current_embedding_l2_norm is not None:
            self.embedding_l2_norm_history.append(current_embedding_l2_norm)
            if len(self.embedding_l2_norm_history) > self.history_limit:
                self.embedding_l2_norm_history = self.embedding_l2_norm_history[-self.history_limit :]

        if self.last_prediction is not None and len(self.recent_signals) > 0:
            prev_prediction = self.recent_signals[-1].get("prediction")
            if prev_prediction == self.last_prediction:
                self.prediction_streak += 1
            else:
                self.prediction_streak = 1
        else:
            self.prediction_streak = 1 if self.last_prediction is not None else 0

        if current_is_attack_prediction:
            self.attack_prediction_streak = self.attack_prediction_streak + 1 if self.attack_prediction_streak > 0 else 1
        else:
            self.attack_prediction_streak = 0

        if current_is_suspicious:
            self.suspicious_streak = self.suspicious_streak + 1 if self.suspicious_streak > 0 else 1
        else:
            self.suspicious_streak = 0

        if current_is_anomaly:
            self.anomaly_streak = self.anomaly_streak + 1 if self.anomaly_streak > 0 else 1
        else:
            self.anomaly_streak = 0

        signal_snapshot = self._build_signal_snapshot(row)
        self.recent_signals.append(signal_snapshot)
        if len(self.recent_signals) > self.history_limit:
            self.recent_signals = self.recent_signals[-self.history_limit :]

        if delta is not None:
            delta["anomaly_detected"] = current_is_anomaly
            delta["suspicious_detected"] = current_is_suspicious
            delta["attack_prediction"] = current_is_attack_prediction
            self.recent_deltas.append(delta)
            if len(self.recent_deltas) > self.history_limit:
                self.recent_deltas = self.recent_deltas[-self.history_limit :]

        return self.snapshot()

    def _build_delta(self, current_row: dict[str, Any]) -> dict[str, Any] | None:
        if self.last_signal_row is None:
            return None

        current_probability = _safe_float(current_row.get("attack_probability"))
        previous_probability = _safe_float(self.last_signal_row.get("attack_probability"))
        current_margin = _safe_float(current_row.get("logit_margin"))
        previous_margin = _safe_float(self.last_signal_row.get("logit_margin"))
        current_embedding = current_row.get("embedding_summary") if isinstance(current_row.get("embedding_summary"), dict) else {}
        previous_embedding = (
            self.last_signal_row.get("embedding_summary")
            if isinstance(self.last_signal_row.get("embedding_summary"), dict)
            else {}
        )
        current_embedding_l2 = _safe_float(current_embedding.get("l2_norm"))
        previous_embedding_l2 = _safe_float(previous_embedding.get("l2_norm"))

        delta = {
            "window_id": current_row.get("window_id"),
            "previous_window_id": self.last_signal_row.get("window_id"),
            "attack_probability_delta": None,
            "logit_margin_delta": None,
            "embedding_l2_norm_delta": None,
            "prediction_changed": None,
            "risk_level_changed": None,
        }

        if current_probability is not None and previous_probability is not None:
            delta["attack_probability_delta"] = current_probability - previous_probability
        if current_margin is not None and previous_margin is not None:
            delta["logit_margin_delta"] = current_margin - previous_margin
        if current_embedding_l2 is not None and previous_embedding_l2 is not None:
            delta["embedding_l2_norm_delta"] = current_embedding_l2 - previous_embedding_l2

        prev_prediction = self.last_signal_row.get("prediction")
        if isinstance(prev_prediction, (int, float)) and isinstance(current_row.get("prediction"), (int, float)):
            delta["prediction_changed"] = int(current_row["prediction"]) != int(prev_prediction)
        prev_risk = self.last_signal_row.get("risk_level")
        curr_risk = current_row.get("risk_level")
        if isinstance(prev_risk, str) and isinstance(curr_risk, str):
            delta["risk_level_changed"] = prev_risk != curr_risk

        return delta

    def _build_signal_snapshot(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "window_id": row.get("window_id"),
            "row_index": row.get("row_index"),
            "prediction": row.get("prediction"),
            "prediction_label": row.get("prediction_label"),
            "attack_probability": row.get("attack_probability"),
            "benign_probability": row.get("benign_probability"),
            "confidence": row.get("confidence"),
            "logit_margin": row.get("logit_margin"),
            "risk_level": row.get("risk_level"),
            "embedding_summary": row.get("embedding_summary"),
            "top_embedding_activations": row.get("top_embedding_activations", [])[:5],
        }

    def snapshot(self) -> dict[str, Any]:
        probability_trend = _trend_label(self.attack_probability_history)
        confidence_trend = _trend_label(self.confidence_history)
        margin_trend = _trend_label(self.logit_margin_history)
        embedding_l2_norm_trend = _trend_label(self.embedding_l2_norm_history)
        last_delta = self.recent_deltas[-1] if self.recent_deltas else None
        last_embedding_delta = None
        if last_delta is not None:
            last_embedding_delta = {
                "window_id": last_delta.get("window_id"),
                "previous_window_id": last_delta.get("previous_window_id"),
                "embedding_l2_norm_delta": last_delta.get("embedding_l2_norm_delta"),
            }
        return {
            "window_count": int(self.window_count),
            "last_window_id": self.last_window_id,
            "attack_probability_ema": self.attack_probability_ema,
            "confidence_ema": self.confidence_ema,
            "logit_margin_ema": self.logit_margin_ema,
            "prediction_streak": int(self.prediction_streak),
            "suspicious_streak": int(self.suspicious_streak),
            "anomaly_streak": int(self.anomaly_streak),
            "attack_prediction_streak": int(self.attack_prediction_streak),
            "last_prediction": self.last_prediction,
            "attack_probability_history": list(self.attack_probability_history),
            "confidence_history": list(self.confidence_history),
            "logit_margin_history": list(self.logit_margin_history),
            "risk_level_history": list(self.risk_level_history),
            "embedding_l2_norm_history": list(self.embedding_l2_norm_history),
            "attack_probability_trend": probability_trend,
            "confidence_trend": confidence_trend,
            "logit_margin_trend": margin_trend,
            "embedding_l2_norm_trend": embedding_l2_norm_trend,
            "rolling_attack_probability_mean": _rolling_mean(self.attack_probability_history),
            "rolling_confidence_mean": _rolling_mean(self.confidence_history),
            "rolling_logit_margin_mean": _rolling_mean(self.logit_margin_history),
            "rolling_embedding_l2_norm_mean": _rolling_mean(self.embedding_l2_norm_history),
            "recent_signals": list(self.recent_signals),
            "last_signal": self.last_signal_row,
            "last_delta": last_delta,
            "last_embedding_delta": last_embedding_delta,
        }
