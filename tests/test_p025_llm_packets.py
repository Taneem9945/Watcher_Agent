from src.mixed_data.events import NormalizedEvent
from src.mixed_data.llm_packets import assert_no_forbidden_keys, build_p025_llm_packet
from src.mixed_data.windowing import build_event_window


def _event(
    event_id: str,
    source_type: str,
    event_type: str,
    uid: str | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=event_id,
        timestamp="2025-07-20T22:11:47Z",
        timestamp_epoch=1753049507.0,
        source_id=source_type,
        source_type=source_type,
        parser="zeek_json",
        role="http_activity" if source_type == "zeek_http" else "network_connection",
        host=None,
        src_ip="128.16.11.13",
        dst_ip="114.0.194.2",
        src_port=52228,
        dst_port=8000,
        username=None,
        uid=uid,
        event_type=event_type,
        summary=f"{event_type} from 128.16.11.13 to 114.0.194.2:8000",
        features={"src_ip_present": 1, "dst_ip_present": 1},
        stable_ids={"uid": uid, "src_ip": "128.16.11.13", "dst_ip": "114.0.194.2"},
        available_fields=["src_ip", "dst_ip", "src_port", "dst_port", "uid"],
        missing_fields=["username"],
    )


def test_build_p025_llm_packet_includes_stable_ids_relationships_and_missingness():
    window = build_event_window(
        window_id=4,
        time_bucket_index=10,
        start_epoch=1753049460.0,
        window_seconds=60,
        events=[
            _event("conn", "zeek_conn", "connection_http", uid="C1"),
            _event("http", "zeek_http", "http_request", uid="C1"),
        ],
    ).to_dict()
    signal = {
        "window_id": 4,
        "signal_type": "mamba_encoder_representation",
        "embedding_summary": {"mean": 0.1, "std": 0.2, "l2_norm": 0.5},
        "top_embedding_activations": [{"index": 1, "value": 0.4, "magnitude": 0.4}],
    }

    packet = build_p025_llm_packet(window, signal)

    assert packet["window_metadata"]["window_id"] == 4
    assert packet["stable_identifiers"]["src_ips"] == ["128.16.11.13"]
    assert packet["stable_identifiers"]["uids"] == ["C1"]
    assert packet["missing_data"]["missing_field_counts"]["username"] == 2
    assert packet["source_meaning"]["source_counts"] == {"zeek_conn": 1, "zeek_http": 1}
    assert packet["relationship_hints"][0]["relationship"] == "shared_uid"
    assert packet["mamba_encoder_signal"]["signal_type"] == "mamba_encoder_representation"
    assert packet["constraints"]["do_not_treat_mamba_signal_as_verdict"] is True
    assert_no_forbidden_keys(packet)


def test_build_p025_llm_packet_stream_memory_tracks_embedding_history():
    window = build_event_window(
        window_id=2,
        time_bucket_index=2,
        start_epoch=100.0,
        window_seconds=60,
        events=[_event("conn", "zeek_conn", "connection_http", uid="C1")],
    ).to_dict()
    signal = {
        "window_id": 2,
        "signal_type": "mamba_encoder_representation",
        "embedding_summary": {"l2_norm": 0.8},
        "top_embedding_activations": [],
    }
    previous = [
        {
            "window_id": 1,
            "signal_type": "mamba_encoder_representation",
            "embedding_summary": {"l2_norm": 0.5},
            "top_embedding_activations": [],
        }
    ]

    packet = build_p025_llm_packet(window, signal, previous_encoder_signals=previous)

    assert packet["stream_memory"]["previous_window_id"] == 1
    assert packet["stream_memory"]["embedding_l2_norm_trend"] == "rising"
    assert packet["stream_memory"]["embedding_l2_norm_delta_from_previous"] == 0.30000000000000004
