from src.mixed_data.events import NormalizedEvent
from src.mixed_data.windowing import create_time_windows, floor_to_window_start


def _event(
    event_id: str,
    timestamp_epoch: float,
    source_type: str,
    event_type: str,
    stable_ids: dict | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=event_id,
        timestamp="2025-07-20T00:00:00Z",
        timestamp_epoch=timestamp_epoch,
        source_id=source_type,
        source_type=source_type,
        parser="test",
        role="test",
        host=None,
        src_ip=None,
        dst_ip=None,
        src_port=None,
        dst_port=None,
        username=None,
        uid=None,
        event_type=event_type,
        summary=event_type,
        features={},
        stable_ids=stable_ids or {},
        available_fields=[],
        missing_fields=[],
    )


def test_floor_to_window_start_uses_epoch_boundary():
    assert floor_to_window_start(1753049507.4, 60) == 1753049460.0


def test_create_time_windows_groups_mixed_events():
    events = [
        _event("late", 1753049518.0, "zeek_http", "http_request", {"uid": "B"}),
        _event("early", 1753049507.0, "zeek_conn", "connection_http", {"uid": "A"}),
        _event("auth", 1753049512.0, "host_auth", "session_opened", {"host": "sjs08a"}),
    ]

    windows = create_time_windows(events, window_seconds=60)

    assert len(windows) == 1
    assert windows[0].event_count == 3
    assert windows[0].source_counts == {"host_auth": 1, "zeek_conn": 1, "zeek_http": 1}
    assert windows[0].event_type_counts == {"connection_http": 1, "http_request": 1, "session_opened": 1}
    assert windows[0].stable_ids == {"host": ["sjs08a"], "uid": ["A", "B"]}
    assert [event["event_id"] for event in windows[0].events] == ["early", "auth", "late"]


def test_create_time_windows_can_include_empty_windows():
    events = [
        _event("first", 120.0, "host_auth", "session_opened"),
        _event("third", 240.0, "zeek_conn", "connection"),
    ]

    windows = create_time_windows(events, window_seconds=60, include_empty=True)

    assert [window.window_id for window in windows] == [0, 1, 2]
    assert [window.event_count for window in windows] == [1, 0, 1]
