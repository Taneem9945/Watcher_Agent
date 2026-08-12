from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .events import NormalizedEvent


@dataclass(frozen=True)
class EventWindow:
    window_id: int
    time_bucket_index: int
    start_timestamp: str
    end_timestamp: str
    start_epoch: float
    end_epoch: float
    event_count: int
    source_counts: dict[str, int]
    event_type_counts: dict[str, int]
    stable_ids: dict[str, list[Any]]
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso_from_epoch(timestamp_epoch: float) -> str:
    return datetime.fromtimestamp(float(timestamp_epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def floor_to_window_start(timestamp_epoch: float, window_seconds: int) -> float:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    return float(int(timestamp_epoch // window_seconds) * window_seconds)


def _collect_stable_ids(events: Sequence[NormalizedEvent]) -> dict[str, list[Any]]:
    collected: dict[str, set[Any]] = {}
    for event in events:
        for key, value in event.stable_ids.items():
            if value is None or value == "":
                continue
            collected.setdefault(key, set()).add(value)
    return {key: sorted(values)[:20] for key, values in sorted(collected.items())}


def build_event_window(
    window_id: int,
    time_bucket_index: int,
    start_epoch: float,
    window_seconds: int,
    events: Sequence[NormalizedEvent],
) -> EventWindow:
    end_epoch = start_epoch + window_seconds
    source_counts = Counter(event.source_type for event in events)
    event_type_counts = Counter(event.event_type for event in events)
    return EventWindow(
        window_id=int(window_id),
        time_bucket_index=int(time_bucket_index),
        start_timestamp=_iso_from_epoch(start_epoch),
        end_timestamp=_iso_from_epoch(end_epoch),
        start_epoch=float(start_epoch),
        end_epoch=float(end_epoch),
        event_count=len(events),
        source_counts=dict(sorted(source_counts.items())),
        event_type_counts=dict(sorted(event_type_counts.items())),
        stable_ids=_collect_stable_ids(events),
        events=[event.to_dict() for event in events],
    )


def create_time_windows(
    events: Iterable[NormalizedEvent],
    window_seconds: int,
    include_empty: bool = False,
) -> list[EventWindow]:
    sorted_events = sorted(events, key=lambda event: (event.timestamp_epoch, event.source_id, event.event_id))
    if not sorted_events:
        return []

    anchor = floor_to_window_start(sorted_events[0].timestamp_epoch, window_seconds)
    buckets: dict[int, list[NormalizedEvent]] = {}
    for event in sorted_events:
        window_id = int((event.timestamp_epoch - anchor) // window_seconds)
        buckets.setdefault(window_id, []).append(event)

    if include_empty:
        window_ids = range(0, max(buckets) + 1)
    else:
        window_ids = sorted(buckets)

    windows = []
    for emitted_window_id, time_bucket_index in enumerate(window_ids):
        start_epoch = anchor + (time_bucket_index * window_seconds)
        windows.append(
            build_event_window(
                window_id=emitted_window_id,
                time_bucket_index=time_bucket_index,
                start_epoch=start_epoch,
                window_seconds=window_seconds,
                events=buckets.get(time_bucket_index, []),
            )
        )
    return windows
