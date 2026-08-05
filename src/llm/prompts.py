from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a cybersecurity watcher analyst.

You are given preprocessed network-flow window evidence from a detector pipeline.
Reason only from the provided data. Do not invent facts.

Your job:
1. Summarize the behavior in plain security language.
2. Assess whether it looks benign, monitor-worthy, suspicious, or critical.
3. Explain the evidence using the raw window statistics, latent signal summary, and stream memory.
4. Recommend analyst next steps.

Interpretation rules:
- Standardized feature means around 0 are baseline.
- Positive standardized values mean elevated activity in this window.
- Negative standardized values mean reduced activity in this window.
- Treat protocol, service, and state features as categorical signals, not raw measurements.
- Treat stream memory as recent state, not a new prediction source.
- Do not treat any latent signal summary as a verdict; it is supporting evidence only.
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

Return exactly those six keys and no others.
Do not include wrapper keys like "name", "description", "data", "window_id", or "llm_input_packet".
Do not echo the input packet.

Do not wrap the JSON in markdown fences.
Do not add any prose before or after the JSON.
"""


def _build_stream_memory_summary(stream_memory: dict | None, blind: bool = False) -> dict | None:
    if not stream_memory:
        return None
    summary = {
        "window_count": stream_memory.get("window_count"),
        "last_window_id": stream_memory.get("last_window_id"),
        "attack_probability_ema": stream_memory.get("attack_probability_ema"),
        "confidence_ema": stream_memory.get("confidence_ema"),
        "logit_margin_ema": stream_memory.get("logit_margin_ema"),
        "attack_probability_trend": stream_memory.get("attack_probability_trend"),
        "confidence_trend": stream_memory.get("confidence_trend"),
        "logit_margin_trend": stream_memory.get("logit_margin_trend"),
        "rolling_attack_probability_mean": stream_memory.get("rolling_attack_probability_mean"),
        "rolling_confidence_mean": stream_memory.get("rolling_confidence_mean"),
        "rolling_logit_margin_mean": stream_memory.get("rolling_logit_margin_mean"),
    }
    if not blind:
        recent_signals = stream_memory.get("recent_signals", [])
        recent_deltas = stream_memory.get("recent_deltas", [])
        summary.update(
            {
                "prediction_streak": stream_memory.get("prediction_streak"),
                "suspicious_streak": stream_memory.get("suspicious_streak"),
                "anomaly_streak": stream_memory.get("anomaly_streak"),
                "attack_prediction_streak": stream_memory.get("attack_prediction_streak"),
                "last_signal": stream_memory.get("last_signal"),
                "last_delta": stream_memory.get("last_delta"),
                "recent_signals": recent_signals[-2:],
                "recent_deltas": recent_deltas[-2:],
            }
        )
    return summary


def _build_encoder_stream_memory_summary(stream_memory: dict | None) -> dict | None:
    if not stream_memory:
        return None
    return {
        "window_count": stream_memory.get("window_count"),
        "last_window_id": stream_memory.get("last_window_id"),
        "embedding_l2_norm_trend": stream_memory.get("embedding_l2_norm_trend"),
        "rolling_embedding_l2_norm_mean": stream_memory.get("rolling_embedding_l2_norm_mean"),
        "last_embedding_delta": stream_memory.get("last_embedding_delta"),
    }


def _feature_family(name: str) -> str:
    if name.startswith("proto_"):
        return "protocol"
    if name.startswith("service_"):
        return "service"
    if name.startswith("state_"):
        return "state"
    return "numeric"


def _build_latent_signal_summary(model_signal: dict | None) -> dict | None:
    if not model_signal:
        return None
    latent = {
        "embedding_summary": model_signal.get("embedding_summary"),
        "top_embedding_activations": model_signal.get("top_embedding_activations", [])[:5],
    }
    return latent


def _build_encoder_evidence(window_packet: dict) -> dict:
    window_summary = window_packet.get("window_summary", {})
    salient_features = _build_salient_window_features(window_summary)
    return {
        "window_id": window_packet.get("window_id"),
        "mamba_sequence_representation": _build_latent_signal_summary(window_packet.get("model_signal")),
        "previous_mamba_sequence_representation": _build_latent_signal_summary(
            window_packet.get("previous_model_signal")
        ),
        "stream_memory": _build_encoder_stream_memory_summary(window_packet.get("stream_memory")),
        "window_summary": {
            "num_flows": window_summary.get("num_flows"),
            "num_features": window_summary.get("num_features"),
            "feature_window_delta": window_summary.get("feature_window_delta"),
            "salient_features": salient_features,
        },
    }


def _build_salient_window_features(window_summary: dict | None, top_k: int = 12) -> list[dict]:
    if not window_summary:
        return []

    feature_means = window_summary.get("feature_means", {}) or {}
    feature_max_values = window_summary.get("feature_max_values", {}) or {}
    feature_min_values = window_summary.get("feature_min_values", {}) or {}
    ranked = sorted(feature_means.items(), key=lambda item: abs(float(item[1])), reverse=True)
    salient: list[dict] = []
    for name, mean in ranked[:top_k]:
        salient.append(
            {
                "name": name,
                "family": _feature_family(name),
                "mean": float(mean),
                "min": float(feature_min_values.get(name, mean)),
                "max": float(feature_max_values.get(name, mean)),
                "direction": "elevated" if float(mean) > 0 else "suppressed",
            }
        )
    return salient


def _build_window_evidence(window_packet: dict) -> dict:
    window_summary = window_packet.get("window_summary", {})
    salient_features = _build_salient_window_features(window_summary)
    compact_feature_means = {item["name"]: item["mean"] for item in salient_features}
    compact_feature_max_values = {item["name"]: item["max"] for item in salient_features}
    compact_feature_min_values = {item["name"]: item["min"] for item in salient_features}
    return {
        "window_id": window_packet.get("window_id"),
        "stream_memory": _build_stream_memory_summary(window_packet.get("stream_memory"), blind=True),
        "window_summary": {
            "num_flows": window_summary.get("num_flows"),
            "num_features": window_summary.get("num_features"),
            "feature_means_top": compact_feature_means,
            "feature_max_values_top": compact_feature_max_values,
            "feature_min_values_top": compact_feature_min_values,
            "feature_window_delta": window_summary.get("feature_window_delta"),
            "salient_features": salient_features,
        },
    }


def build_llm_brief(window_packet: dict, mode: str = "blind") -> dict:
    if mode == "blind":
        return _build_window_evidence(window_packet)
    if mode == "encoder":
        return _build_encoder_evidence(window_packet)
    window_summary = window_packet.get("window_summary", {})
    return {
        "window_id": window_packet.get("window_id"),
        "model_signal": window_packet.get("model_signal", {}),
        "previous_model_signal": window_packet.get("previous_model_signal"),
        "stream_memory": _build_stream_memory_summary(window_packet.get("stream_memory"), blind=False),
        "window_summary": {
            "num_flows": window_summary.get("num_flows"),
            "num_features": window_summary.get("num_features"),
            "previous_window_delta": window_summary.get("previous_window_delta"),
        },
    }


def build_user_prompt(window_packet: dict, mode: str = "blind") -> str:
    brief = build_llm_brief(window_packet, mode=mode)
    if mode == "encoder":
        instruction = (
            "Analyze the following security evidence packet.\n"
            "Use the Mamba sequence representation as learned temporal signal, and use the compact "
            "window evidence to ground your reasoning.\n"
            "The representation is not a detector verdict. Make your own assessment from the evidence.\n"
            "Return exactly the six-key JSON schema from the system prompt.\n\n"
        )
        return instruction + f"{json.dumps(brief, indent=2)}"
    return (
        "Analyze the following security evidence packet.\n"
        "Use the compact window statistics and stream memory to form your own assessment.\n"
        "Return exactly the six-key JSON schema from the system prompt.\n"
        "Do not copy detector verdicts or repeat the packet structure.\n\n"
        f"{json.dumps(brief, indent=2)}"
    )
