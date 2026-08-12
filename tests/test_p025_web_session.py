import asyncio
import json

import src.mixed_data.web_session as web_session
from src.mixed_data.web_session import P025WebReplaySession


def test_p025_web_replay_session_emits_fresh_encoder_only_record(tmp_path, monkeypatch):
    packets_path = tmp_path / "packets.jsonl"
    assessments_path = tmp_path / "assessments.jsonl"
    packet = {
        "packet_type": "p025_mixed_window_llm_evidence",
        "window_metadata": {"window_id": 0, "event_count": 2, "source_counts": {"host_auth": 2}},
        "stable_identifiers": {"hosts": ["sjs08a"]},
        "missing_data": {"missing_field_counts": {"src_ip": 2}},
        "mamba_encoder_signal": {
            "signal_type": "mamba_encoder_representation",
            "embedding_summary": {"l2_norm": 0.5},
            "top_embedding_activations": [],
        },
        "stream_memory": {"embedding_l2_norm_trend": "insufficient"},
    }
    stale_assessment = {
        "window_id": 0,
        "assessment": "stale",
        "severity": "low",
        "summary": "This should not be reused.",
        "evidence": ["Cached result."],
        "recommended_actions": [],
        "analyst_note": "",
    }
    packets_path.write_text(json.dumps(packet) + "\n", encoding="utf-8")
    assessments_path.write_text(json.dumps(stale_assessment) + "\n", encoding="utf-8")

    calls = []

    def fake_assess_p025_packet(client, incoming_packet):
        calls.append(incoming_packet)
        return {
            "window_id": 0,
            "assessment": "benign",
            "severity": "low",
            "summary": "Fresh routine auth activity.",
            "evidence": ["Fresh assessment was generated."],
            "recommended_actions": [],
            "analyst_note": "",
        }

    monkeypatch.setattr(web_session, "assess_p025_packet", fake_assess_p025_packet)

    session = P025WebReplaySession(
        packets_path=str(packets_path),
        assessments_path=str(assessments_path),
        max_windows=1,
    )

    records = asyncio.run(session.step_row())

    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "p025_encoder_llm_window"
    assert record["window"] == packet["window_metadata"]
    assert "mamba_encoder_signal" not in record
    assert record["llm_packet"] == packet
    assert record["llm_packet"]["mamba_encoder_signal"]["signal_type"] == "mamba_encoder_representation"
    assert record["llm_assessment"]["assessment"] == "benign"
    assert record["assessment_source"] == "ollama_live"
    assert record["run_id"] == session.run_id
    assert calls == [packet]
    assert "mamba_signal_packet" not in record
    assert "assessment_cached" not in record
    assert "prediction" not in json.dumps(record)
    assert "attack_probability" not in json.dumps(record)
    assert session.status()["stream_mode"] == "p025_encoder_llm"
    assert session.status()["run_id"] == session.run_id


def test_p025_web_replay_session_rejects_classifier_fields(tmp_path):
    packets_path = tmp_path / "packets.jsonl"
    packet = {
        "packet_type": "p025_mixed_window_llm_evidence",
        "window_metadata": {"window_id": 0, "event_count": 1},
        "mamba_encoder_signal": {
            "signal_type": "mamba_encoder_representation",
            "embedding_summary": {"l2_norm": 0.5},
            "prediction": 0,
        },
    }
    packets_path.write_text(json.dumps(packet) + "\n", encoding="utf-8")

    try:
        P025WebReplaySession(packets_path=str(packets_path))
    except ValueError as exc:
        assert "forbidden keys" in str(exc)
    else:
        raise AssertionError("expected classifier fields to be rejected")


def test_p025_web_replay_session_rejects_non_p025_packets(tmp_path):
    packets_path = tmp_path / "packets.jsonl"
    packet = {
        "window_id": 0,
        "context_packet": {"source": "old_unsw_pipeline"},
    }
    packets_path.write_text(json.dumps(packet) + "\n", encoding="utf-8")

    try:
        P025WebReplaySession(packets_path=str(packets_path))
    except ValueError as exc:
        assert "only accepts P025 mixed-source packets" in str(exc)
    else:
        raise AssertionError("expected non-P025 packets to be rejected")
