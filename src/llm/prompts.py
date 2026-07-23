from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a cybersecurity watcher analyst.

You are given preprocessed network-flow window data from a detector pipeline.
Reason only from the provided data. Do not invent facts.

Your job:
1. Summarize the behavior in the window in plain security language.
2. Assess whether it looks benign, monitor-worthy, suspicious, or critical.
3. Explain the evidence using the model signal, stream memory, and window statistics.
4. Recommend analyst next steps.

Interpretation rules:
- Standardized feature means around 0 are baseline.
- Positive standardized values mean elevated activity in this window.
- Negative standardized values mean reduced activity in this window.
- Treat protocol, service, and state features as categorical signals, not raw measurements.
- Treat stream memory as recent state, not a new prediction source.
- Do not describe the data as "packets" if the input is a window summary unless the window explicitly says so.
- Do not invent attack names, exploits, or vendor-specific indicators that are not supported by the input.

Return valid JSON only with this schema:
{
  "assessment": "benign|monitor|suspicious|critical",
  "severity": "low|medium|high|critical",
  "summary": "...",
  "evidence": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "analyst_note": "..."
}

Do not wrap the JSON in markdown fences.
Do not add any prose before or after the JSON.
"""


def _build_stream_memory_summary(stream_memory: dict | None) -> dict | None:
    if not stream_memory:
        return None
    recent_signals = stream_memory.get("recent_signals", [])
    recent_deltas = stream_memory.get("recent_deltas", [])
    return {
        "window_count": stream_memory.get("window_count"),
        "last_window_id": stream_memory.get("last_window_id"),
        "attack_probability_ema": stream_memory.get("attack_probability_ema"),
        "confidence_ema": stream_memory.get("confidence_ema"),
        "logit_margin_ema": stream_memory.get("logit_margin_ema"),
        "attack_probability_trend": stream_memory.get("attack_probability_trend"),
        "confidence_trend": stream_memory.get("confidence_trend"),
        "logit_margin_trend": stream_memory.get("logit_margin_trend"),
        "prediction_streak": stream_memory.get("prediction_streak"),
        "suspicious_streak": stream_memory.get("suspicious_streak"),
        "anomaly_streak": stream_memory.get("anomaly_streak"),
        "attack_prediction_streak": stream_memory.get("attack_prediction_streak"),
        "rolling_attack_probability_mean": stream_memory.get("rolling_attack_probability_mean"),
        "rolling_confidence_mean": stream_memory.get("rolling_confidence_mean"),
        "rolling_logit_margin_mean": stream_memory.get("rolling_logit_margin_mean"),
        "last_signal": stream_memory.get("last_signal"),
        "last_delta": stream_memory.get("last_delta"),
        "recent_signals": recent_signals[-2:],
        "recent_deltas": recent_deltas[-2:],
    }


def build_llm_brief(window_packet: dict) -> dict:
    window_summary = window_packet.get("window_summary", {})
    return {
        "window_id": window_packet.get("window_id"),
        "model_signal": window_packet.get("model_signal", {}),
        "previous_model_signal": window_packet.get("previous_model_signal"),
        "stream_memory": _build_stream_memory_summary(window_packet.get("stream_memory")),
        "window_summary": {
            "num_flows": window_summary.get("num_flows"),
            "num_features": window_summary.get("num_features"),
            "previous_window_delta": window_summary.get("previous_window_delta"),
        },
    }


def build_user_prompt(window_packet: dict) -> str:
    brief = build_llm_brief(window_packet)
    return (
        "Analyze the following Mamba signal and supporting window summary.\n"
        "Use the model_signal as the primary evidence, stream_memory as recent history, and the window summary as supporting context.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(brief, indent=2)}"
    )
