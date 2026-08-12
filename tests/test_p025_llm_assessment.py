import json

from src.mixed_data.llm_assessment import assess_p025_packet, build_p025_user_prompt


class FakeClient:
    def __init__(self, response):
        self.responses = response if isinstance(response, list) else [response]
        self.messages = None
        self.calls = 0

    def chat_raw(self, messages, stream=False):
        self.messages = messages
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _packet():
    return {
        "packet_type": "p025_mixed_window_llm_evidence",
        "window_metadata": {"window_id": 9, "event_count": 2},
        "stable_identifiers": {"src_ips": ["128.16.11.13"], "hosts": ["sjs08a"]},
        "missing_data": {"missing_field_counts": {"username": 1}},
        "readable_event_evidence": [
            {"summary": "GET /exploits from 128.16.11.13"},
            {"summary": "session opened for user root"},
        ],
        "relationship_hints": [{"relationship": "shared_src_ip", "value": "128.16.11.13"}],
        "mamba_encoder_signal": {
            "signal_type": "mamba_encoder_representation",
            "embedding_summary": {"l2_norm": 0.5},
            "note": "Representation-only signal; this is not a classifier verdict.",
        },
        "stream_memory": {"embedding_l2_norm_trend": "stable"},
    }


def test_build_p025_user_prompt_includes_packet_context():
    prompt = build_p025_user_prompt(_packet())

    assert "stable_identifiers" in prompt
    assert "relationship_hints" in prompt
    assert "mamba_encoder_signal" in prompt
    assert "Required reasoning order" in prompt
    assert "supporting temporal context" in prompt
    assert "Authoritative missing_field_counts" in prompt
    assert "Authoritative stable_identifiers" in prompt
    assert "Authoritative source_meaning" in prompt
    assert '"username": 1' in prompt


def test_assess_p025_packet_returns_window_id_and_input_packet():
    response = json.dumps(
        {
            "assessment": "monitor",
            "severity": "low",
            "summary": "HTTP and auth activity should be watched.",
            "evidence": ["Observed HTTP request and auth session evidence."],
            "recommended_actions": ["Review adjacent windows."],
            "analyst_note": "Mamba signal is supporting context only.",
        }
    )
    client = FakeClient(response)

    assessment = assess_p025_packet(client, _packet())

    assert assessment["window_id"] == 9
    assert assessment["assessment"] == "monitor"
    assert assessment["llm_input_packet"]["window_metadata"]["window_id"] == 9
    assert client.messages[0]["role"] == "system"
    assert client.messages[1]["role"] == "user"
    assert "Never say there are no missing fields unless" in client.messages[0]["content"]
    assert "Mamba/S6 is being used as a sequence encoder" in client.messages[0]["content"]
    assert "Do not classify from the embedding alone" in client.messages[0]["content"]
    assert "prediction, attack_probability, risk_level, and true_label" in client.messages[0]["content"]


def test_assess_p025_packet_repairs_missing_data_contradiction():
    bad_response = json.dumps(
        {
            "assessment": "benign",
            "severity": "low",
            "summary": "There are no missing fields.",
            "evidence": ["No missing IP or port information."],
            "recommended_actions": [],
            "analyst_note": "",
        }
    )
    repaired_response = json.dumps(
        {
            "assessment": "benign",
            "severity": "low",
            "summary": "Routine activity, with username missing according to the packet.",
            "evidence": ["The packet reports username as missing once."],
            "recommended_actions": [],
            "analyst_note": "Missing data was handled according to packet counts.",
        }
    )
    client = FakeClient([bad_response, repaired_response])

    assessment = assess_p025_packet(client, _packet())

    assert client.calls == 2
    assert assessment["summary"] == "Routine activity, with username missing according to the packet."


def test_assess_p025_packet_defaults_missing_analyst_note():
    response = json.dumps(
        {
            "assessment": "monitor",
            "severity": "low",
            "summary": "HTTP activity observed.",
            "evidence": ["One HTTP event."],
            "recommended_actions": ["Review adjacent windows."],
        }
    )
    client = FakeClient(response)

    assessment = assess_p025_packet(client, _packet())

    assert assessment["analyst_note"] == ""
    assert assessment["assessment"] == "monitor"


def test_assess_p025_packet_unwraps_nested_assessment():
    response = json.dumps(
        {
            "result": {
                "assessment": "benign",
                "severity": "low",
                "summary": "Routine activity.",
                "evidence": "Single routine event.",
                "recommended_actions": "Continue monitoring.",
            }
        }
    )
    client = FakeClient(response)

    assessment = assess_p025_packet(client, _packet())

    assert assessment["assessment"] == "benign"
    assert assessment["evidence"] == ["Single routine event."]
    assert assessment["recommended_actions"] == ["Continue monitoring."]
