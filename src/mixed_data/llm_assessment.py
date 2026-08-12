from __future__ import annotations

import json
from typing import Any

from src.llm.ollama_client import OllamaClient, OllamaError
from src.llm.ollama_watcher import _safe_json_loads


EXPECTED_P025_ASSESSMENT_KEYS = {
    "assessment",
    "severity",
    "summary",
    "evidence",
    "recommended_actions",
    "analyst_note",
}


P025_SYSTEM_PROMPT = """You are a cybersecurity watcher analyst.

You are given one 60-second mixed-source security evidence packet.
Reason only from the provided data. Do not invent facts.

Current architecture:
- Mamba/S6 is being used as a sequence encoder/signal producer.
- The LLM is the assessment and recommendation layer.
- The packet intentionally excludes classifier answers such as prediction, attack_probability, risk_level, and true_label.
- The current encoder signal may be untrained or only partially trained, so treat it as supporting temporal signal, not authoritative security evidence.

The packet may include:
- readable event evidence from sources such as Zeek and host auth logs
- stable identifiers such as IPs, hosts, usernames, and connection UIDs
- missing data summaries
- relationship hints
- Mamba encoder representation summaries
- recent stream memory

Important rules:
- The Mamba encoder signal is not a classifier verdict.
- Do not treat embedding values as direct proof of attack.
- Do not classify from the embedding alone. Use readable event evidence and stable identifiers first.
- Use stable identifiers and readable event evidence to ground your reasoning.
- Preserve stable facts exactly. Do not rewrite, merge, or invent IPs, hosts, usernames, UIDs, or source types.
- Use relationship_hints to explain possible links, such as shared UID between Zeek connection and HTTP events, or repeated host/user activity.
- Missing data can be useful context, but do not overstate it.
- When discussing missing data, use the packet's missing_data section exactly.
- Never say there are no missing fields unless missing_data.missing_field_counts is empty.
- Treat source_type as source meaning. For example, host_auth and zeek_http are different evidence types.
- Use stream_memory only as recent context about prior windows, not as a separate detector.
- Do not invent labels, attack names, malware families, CVEs, or dataset answers.
- If evidence is routine or weak, say monitor or benign instead of exaggerating.
- Prefer monitor over suspicious when the only unusual item is an embedding change without readable supporting evidence.

Return valid JSON only with this exact schema:
{
  "assessment": "benign|monitor|suspicious|critical",
  "severity": "low|medium|high|critical",
  "summary": "...",
  "evidence": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "analyst_note": "..."
}

Return exactly those six keys and no others.
Do not echo the input packet.
Do not wrap the JSON in markdown fences.
Do not add any prose before or after the JSON.
"""


def build_p025_user_prompt(packet: dict[str, Any]) -> str:
    missing_counts = packet.get("missing_data", {}).get("missing_field_counts", {})
    stable_identifiers = packet.get("stable_identifiers", {})
    source_meaning = packet.get("source_meaning", {})
    return (
        "Assess this P025 mixed-source watcher packet.\n"
        "Required reasoning order:\n"
        "1. Start from readable_event_evidence and source_meaning.\n"
        "2. Ground the analysis with stable_identifiers.\n"
        "3. Use relationship_hints to connect events when justified.\n"
        "4. Use missing_data exactly as provided.\n"
        "5. Use mamba_encoder_signal and stream_memory only as supporting temporal context.\n"
        "6. Make your own assessment without classifier labels, probabilities, risk levels, or true labels.\n\n"
        f"Authoritative source_meaning for this packet: {json.dumps(source_meaning, sort_keys=True)}.\n"
        f"Authoritative stable_identifiers for this packet: {json.dumps(stable_identifiers, sort_keys=True)}.\n"
        f"Authoritative missing_field_counts for this packet: {json.dumps(missing_counts, sort_keys=True)}.\n"
        "If that object is not empty, do not state or imply that there are no missing fields.\n\n"
        f"{json.dumps(packet, indent=2)}"
    )


def looks_like_p025_assessment(payload: dict[str, Any]) -> bool:
    return EXPECTED_P025_ASSESSMENT_KEYS.issubset(payload.keys())


def _unwrap_assessment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if looks_like_p025_assessment(payload):
        return payload
    for key in ("llm_assessment", "assessment_json", "result", "response"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            unwrapped = _unwrap_assessment_payload(nested)
            if looks_like_p025_assessment(unwrapped) or {"assessment", "severity", "summary"}.issubset(unwrapped.keys()):
                return unwrapped
    return payload


def _normalize_assessment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    assessment = _unwrap_assessment_payload(payload)
    if "analyst_note" not in assessment:
        assessment["analyst_note"] = ""
    for key in ("evidence", "recommended_actions"):
        value = assessment.get(key)
        if value is None:
            assessment[key] = []
        elif isinstance(value, str):
            assessment[key] = [value]
    return assessment


def _text_fields_for_validation(assessment: dict[str, Any]) -> str:
    parts = [
        str(assessment.get("summary", "")),
        str(assessment.get("analyst_note", "")),
    ]
    evidence = assessment.get("evidence")
    if isinstance(evidence, list):
        parts.extend(str(item) for item in evidence)
    actions = assessment.get("recommended_actions")
    if isinstance(actions, list):
        parts.extend(str(item) for item in actions)
    return " ".join(parts).lower()


def _contradicts_missing_data(packet: dict[str, Any], assessment: dict[str, Any]) -> bool:
    missing_counts = packet.get("missing_data", {}).get("missing_field_counts", {})
    if not missing_counts:
        return False
    text = _text_fields_for_validation(assessment)
    contradiction_phrases = (
        "no missing",
        "without missing",
        "missing fields: none",
        "no fields missing",
    )
    return any(phrase in text for phrase in contradiction_phrases)


def assess_p025_packet(
    client: OllamaClient,
    packet: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": P025_SYSTEM_PROMPT},
        {"role": "user", "content": build_p025_user_prompt(packet)},
    ]
    response_text = client.chat_raw(messages=messages, stream=False)
    assessment = _normalize_assessment_payload(_safe_json_loads(response_text))
    if not looks_like_p025_assessment(assessment):
        repair_messages = [
            {
                "role": "system",
                "content": (
                    P025_SYSTEM_PROMPT
                    + "\nThe previous answer was invalid. Rewrite it as the required six-key JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite this response into the exact required six-key assessment JSON schema.\n\n"
                    f"Original response:\n{response_text}\n\n"
                    f"Evidence packet:\n{json.dumps(packet, indent=2)}"
                ),
            },
        ]
        response_text = client.chat_raw(messages=repair_messages, stream=False)
        assessment = _normalize_assessment_payload(_safe_json_loads(response_text))
    if _contradicts_missing_data(packet, assessment):
        missing_counts = packet.get("missing_data", {}).get("missing_field_counts", {})
        repair_messages = [
            {
                "role": "system",
                "content": (
                    P025_SYSTEM_PROMPT
                    + "\nThe previous answer contradicted the packet's missing_data section. "
                    + "Correct the assessment JSON while preserving the six-key schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The packet has these missing_field_counts: {json.dumps(missing_counts, sort_keys=True)}.\n"
                    "The previous response said or implied there were no missing fields. "
                    "Rewrite the assessment so missing-data statements match those counts exactly.\n\n"
                    f"Previous response:\n{json.dumps(assessment, indent=2)}\n\n"
                    f"Evidence packet:\n{json.dumps(packet, indent=2)}"
                ),
            },
        ]
        response_text = client.chat_raw(messages=repair_messages, stream=False)
        assessment = _normalize_assessment_payload(_safe_json_loads(response_text))
    if not looks_like_p025_assessment(assessment):
        raise OllamaError(f"Ollama assessment was missing required keys: {sorted(assessment.keys())}")
    if _contradicts_missing_data(packet, assessment):
        raise OllamaError("Ollama assessment contradicted packet missing_data after repair")
    assessment["window_id"] = int(packet.get("window_metadata", {}).get("window_id", -1))
    assessment["llm_input_packet"] = packet
    assessment["raw_response_text"] = response_text
    return assessment
