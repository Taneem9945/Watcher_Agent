import numpy as np

from src.mixed_data.events import NormalizedEvent
from src.mixed_data.features import build_feature_names, vectorize_event, windows_to_tensor_batch
from src.mixed_data.windowing import build_event_window


def _event(
    event_id: str,
    timestamp_epoch: float,
    source_type: str,
    event_type: str,
    features: dict | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=event_id,
        timestamp="2025-07-20T00:00:00Z",
        timestamp_epoch=timestamp_epoch,
        source_id=source_type,
        source_type=source_type,
        parser="zeek_json" if source_type.startswith("zeek") else "linux_auth_text",
        role="network_connection" if source_type == "zeek_conn" else "host_authentication",
        host="sjs08a" if source_type == "host_auth" else None,
        src_ip="10.0.0.1" if source_type.startswith("zeek") else None,
        dst_ip="10.0.0.2" if source_type.startswith("zeek") else None,
        src_port=12345 if source_type.startswith("zeek") else None,
        dst_port=80 if source_type.startswith("zeek") else None,
        username="root" if source_type == "host_auth" else None,
        uid="C1" if source_type.startswith("zeek") else None,
        event_type=event_type,
        summary=event_type,
        features=features or {},
        stable_ids={},
        available_fields=[],
        missing_fields=["username"] if source_type.startswith("zeek") else ["src_ip", "dst_ip"],
    )


def test_build_feature_names_includes_base_categorical_and_event_features():
    window = build_event_window(
        window_id=0,
        time_bucket_index=0,
        start_epoch=100.0,
        window_seconds=60,
        events=[
            _event("a", 101.0, "zeek_conn", "connection_http", {"duration": 0.2}),
            _event("b", 102.0, "host_auth", "session_opened", {"event_session_opened": 1}),
        ],
    )

    names = build_feature_names([window])

    assert "offset_seconds" in names
    assert "source_type=host_auth" in names
    assert "source_type=zeek_conn" in names
    assert "event_type=connection_http" in names
    assert "feature.duration" in names
    assert "feature.event_session_opened" in names


def test_vectorize_event_sets_one_hot_and_numeric_values():
    event = _event("a", 112.0, "zeek_conn", "connection_http", {"duration": 0.2})
    window = build_event_window(
        window_id=0,
        time_bucket_index=0,
        start_epoch=100.0,
        window_seconds=60,
        events=[event],
    )
    names = build_feature_names([window])

    vector = vectorize_event(event.to_dict(), window, names)

    assert vector[names.index("offset_seconds")] == 12.0
    assert vector[names.index("src_ip_present")] == 1.0
    assert vector[names.index("source_type=zeek_conn")] == 1.0
    assert vector[names.index("event_type=connection_http")] == 1.0
    assert vector[names.index("feature.duration")] == np.float32(0.2)


def test_windows_to_tensor_batch_pads_and_masks():
    window_a = build_event_window(
        window_id=0,
        time_bucket_index=0,
        start_epoch=100.0,
        window_seconds=60,
        events=[
            _event("a", 101.0, "zeek_conn", "connection_http"),
            _event("b", 102.0, "host_auth", "session_opened"),
        ],
    )
    window_b = build_event_window(
        window_id=1,
        time_bucket_index=1,
        start_epoch=160.0,
        window_seconds=60,
        events=[_event("c", 161.0, "zeek_conn", "connection_dns")],
    )

    batch = windows_to_tensor_batch([window_a, window_b], max_events=3, standardize=False)

    assert batch.X.shape == (2, 3, len(batch.feature_names))
    assert batch.mask.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert np.all(batch.X[0, 2] == 0.0)
    assert batch.metadata[0]["event_count"] == 2
    assert batch.metadata[0]["encoded_event_count"] == 2
