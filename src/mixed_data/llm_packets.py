from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


FORBIDDEN_LLM_PACKET_KEYS = {
    "prediction",
    "prediction_label",
    "attack_probability",
    "benign_probability",
    "risk_level",
    "true_label",
}


def _compact_counter(values: Sequence[Any], limit: int = 12) -> dict[str, int]:
    counter = Counter(str(value) for value in values if value not in (None, ""))
    return dict(counter.most_common(limit))


def _unique_values(values: Sequence[Any], limit: int = 20) -> list[Any]:
    seen = []
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def _event_values(events: Sequence[dict[str, Any]], key: str) -> list[Any]:
    return [event.get(key) for event in events]


def build_stable_identifier_summary(events: Sequence[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    stable_from_events: dict[str, list[Any]] = {}
    for event in events:
        for key, value in (event.get("stable_ids") or {}).items():
            if value in (None, ""):
                continue
            stable_from_events.setdefault(key, [])
            if value not in stable_from_events[key]:
                stable_from_events[key].append(value)

    summary = {
        "src_ips": _unique_values(_event_values(events, "src_ip"), limit=limit),
        "dst_ips": _unique_values(_event_values(events, "dst_ip"), limit=limit),
        "hosts": _unique_values(_event_values(events, "host"), limit=limit),
        "usernames": _unique_values(_event_values(events, "username"), limit=limit),
        "uids": _unique_values(_event_values(events, "uid"), limit=limit),
        "source_types": _unique_values(_event_values(events, "source_type"), limit=limit),
        "from_event_stable_ids": {key: values[:limit] for key, values in sorted(stable_from_events.items())},
    }
    return summary


def build_missing_data_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    missing_fields = []
    available_fields = []
    for event in events:
        missing_fields.extend(event.get("missing_fields") or [])
        available_fields.extend(event.get("available_fields") or [])
    return {
        "events_with_missing_fields": sum(1 for event in events if event.get("missing_fields")),
        "missing_field_counts": _compact_counter(missing_fields),
        "available_field_counts": _compact_counter(available_fields),
    }


def build_readable_event_evidence(events: Sequence[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    evidence = []
    for event in events[:limit]:
        evidence.append(
            {
                "timestamp": event.get("timestamp"),
                "source_type": event.get("source_type"),
                "event_type": event.get("event_type"),
                "src_ip": event.get("src_ip"),
                "dst_ip": event.get("dst_ip"),
                "host": event.get("host"),
                "username": event.get("username"),
                "uid": event.get("uid"),
                "summary": event.get("summary"),
            }
        )
    return evidence


def build_relationship_hints(events: Sequence[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    relationship_keys = ("uid", "src_ip", "dst_ip", "host", "username")
    for key in relationship_keys:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for event in events:
            value = event.get(key)
            if value in (None, ""):
                continue
            groups.setdefault(value, []).append(event)

        for value, grouped_events in groups.items():
            source_types = sorted({str(event.get("source_type")) for event in grouped_events})
            event_types = sorted({str(event.get("event_type")) for event in grouped_events})
            if len(grouped_events) < 2 and len(source_types) < 2:
                continue
            hints.append(
                {
                    "relationship": f"shared_{key}",
                    "value": value,
                    "event_count": len(grouped_events),
                    "source_types": source_types,
                    "event_types": event_types,
                    "example_summaries": [event.get("summary") for event in grouped_events[:3]],
                }
            )
            if len(hints) >= limit:
                return hints
    return hints


def build_encoder_stream_memory(
    current_signal: dict[str, Any],
    previous_signals: Sequence[dict[str, Any]],
    history_limit: int = 5,
) -> dict[str, Any]:
    recent = list(previous_signals)[-history_limit:]
    current_embedding = current_signal.get("embedding_summary") or {}
    previous_embedding = (recent[-1].get("embedding_summary") if recent else None) or {}
    current_l2 = current_embedding.get("l2_norm")
    previous_l2 = previous_embedding.get("l2_norm")
    history = [
        signal.get("embedding_summary", {}).get("l2_norm")
        for signal in [*recent, current_signal]
        if isinstance(signal.get("embedding_summary"), dict)
        and isinstance(signal.get("embedding_summary", {}).get("l2_norm"), (int, float))
    ]
    trend = "insufficient"
    if len(history) >= 2:
        if history[-1] > history[0] + 1e-6:
            trend = "rising"
        elif history[-1] < history[0] - 1e-6:
            trend = "falling"
        else:
            trend = "stable"
    return {
        "window_count_including_current": len(previous_signals) + 1,
        "previous_window_id": recent[-1].get("window_id") if recent else None,
        "embedding_l2_norm_history": history[-history_limit:],
        "embedding_l2_norm_trend": trend,
        "embedding_l2_norm_delta_from_previous": (
            None
            if not isinstance(current_l2, (int, float)) or not isinstance(previous_l2, (int, float))
            else float(current_l2) - float(previous_l2)
        ),
        "recent_encoder_signals": [
            {
                "window_id": signal.get("window_id"),
                "embedding_summary": signal.get("embedding_summary"),
                "top_embedding_activations": signal.get("top_embedding_activations", [])[:5],
            }
            for signal in recent
        ],
    }


def build_p025_llm_packet(
    window: dict[str, Any],
    encoder_signal: dict[str, Any],
    previous_encoder_signals: Sequence[dict[str, Any]] | None = None,
    evidence_limit: int = 12,
) -> dict[str, Any]:
    previous_encoder_signals = previous_encoder_signals or []
    events = window.get("events") or []
    return {
        "packet_type": "p025_mixed_window_llm_evidence",
        "task": (
            "Assess whether this 60-second mixed-source security window appears normal, monitor-worthy, "
            "suspicious, or critical. Use readable evidence and the Mamba encoder representation as supporting "
            "sequence signal, not as a detector verdict."
        ),
        "window_metadata": {
            "window_id": window.get("window_id"),
            "time_bucket_index": window.get("time_bucket_index"),
            "start_timestamp": window.get("start_timestamp"),
            "end_timestamp": window.get("end_timestamp"),
            "event_count": window.get("event_count"),
            "source_counts": window.get("source_counts"),
            "event_type_counts": window.get("event_type_counts"),
            "window_seconds": (
                None
                if window.get("start_epoch") is None or window.get("end_epoch") is None
                else float(window["end_epoch"]) - float(window["start_epoch"])
            ),
        },
        "stable_identifiers": build_stable_identifier_summary(events),
        "source_meaning": {
            "source_types_seen": sorted((window.get("source_counts") or {}).keys()),
            "source_counts": window.get("source_counts"),
            "event_type_counts": window.get("event_type_counts"),
        },
        "missing_data": build_missing_data_summary(events),
        "readable_event_evidence": build_readable_event_evidence(events, limit=evidence_limit),
        "relationship_hints": build_relationship_hints(events),
        "mamba_encoder_signal": {
            "signal_type": encoder_signal.get("signal_type"),
            "embedding_summary": encoder_signal.get("embedding_summary"),
            "top_embedding_activations": encoder_signal.get("top_embedding_activations", [])[:8],
            "note": "Representation-only signal; this is not a classifier verdict.",
        },
        "stream_memory": build_encoder_stream_memory(encoder_signal, previous_encoder_signals),
        "constraints": {
            "excluded_fields": sorted(FORBIDDEN_LLM_PACKET_KEYS),
            "do_not_treat_mamba_signal_as_verdict": True,
            "no_dataset_answer_or_true_label_available": True,
        },
    }


def assert_no_forbidden_keys(payload: Any) -> None:
    if isinstance(payload, dict):
        overlap = FORBIDDEN_LLM_PACKET_KEYS.intersection(payload.keys())
        if overlap:
            raise ValueError(f"LLM packet contains forbidden keys: {sorted(overlap)}")
        for value in payload.values():
            assert_no_forbidden_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_forbidden_keys(item)
