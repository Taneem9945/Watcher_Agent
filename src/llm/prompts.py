from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a cybersecurity watcher analyst.

You are given timestamped security-window evidence from a Watcher Agent pipeline.
The pipeline uses Mamba/S6 as a sequence encoder. Mamba produces a learned
sequence representation of recent behavior; it is not the final security verdict.

Reason only from the provided evidence. Do not invent facts.

Your job:
1. Interpret the Mamba sequence representation as supporting temporal signal.
2. Ground that interpretation in readable event evidence and stable facts.
3. Use source meaning, missing data, and relationship hints when they are provided.
4. Assess whether the window looks benign, monitor-worthy, suspicious, or critical.
5. Explain the evidence clearly enough for a security analyst to inspect.
6. Recommend practical analyst next steps.

Interpretation rules:
- The Mamba representation is a learned numeric summary of sequence behavior, not a human-named feature list.
- Large embedding norms, strong top activations, or rising embedding deltas may suggest stronger sequence change, but they are not attacks by themselves.
- Use the readable window evidence to explain why a representation change matters.
- Preserve the distinction between stable facts and learned signal.
- Stable identifiers such as IPs, hosts, usernames, UIDs, and source types are grounding facts, not Mamba outputs.
- Relationship hints such as shared UID, shared source IP, shared host, or shared username can connect events into a possible incident chain.
- Missing data must be interpreted from explicit missing-data fields when they are present.
- Standardized feature means around 0 are baseline.
- Positive standardized values mean elevated activity in this window.
- Negative standardized values mean reduced activity in this window.
- Treat protocol, service, and state features as categorical signals, not raw measurements.
- Treat stream memory as recent history, not a separate detector verdict.
- Do not treat any latent signal summary, classifier signal, or trend value as a final answer.
- Do not infer a dataset label, ground-truth answer, or attack probability unless it is explicitly present and intended for use.
- Do not describe the data as "packets" if the input is a window summary unless the window explicitly says so.
- Do not invent attack names, exploits, or vendor-specific indicators that are not supported by the input.
- If evidence is weak or ambiguous, choose "monitor" and say what would need to be checked next.

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
            "Analyze the following Watcher Agent evidence packet.\n"
            "Use the Mamba sequence representation as learned temporal signal. Use the compact "
            "window evidence to ground what that signal may mean in security terms.\n"
            "Make your own assessment. Do not assume the representation is an attack/benign label.\n"
            "Mention representation changes only when they help explain the readable evidence.\n"
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
