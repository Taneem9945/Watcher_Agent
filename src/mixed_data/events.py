from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedEvent:
    event_id: str
    timestamp: str
    timestamp_epoch: float
    source_id: str
    source_type: str
    parser: str
    role: str
    host: str | None
    src_ip: str | None
    dst_ip: str | None
    src_port: int | None
    dst_port: int | None
    username: str | None
    uid: str | None
    event_type: str
    summary: str
    features: dict[str, Any] = field(default_factory=dict)
    stable_ids: dict[str, Any] = field(default_factory=dict)
    available_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_available_missing(values: dict[str, Any], tracked_fields: list[str]) -> tuple[list[str], list[str]]:
    available = []
    missing = []
    for field_name in tracked_fields:
        value = values.get(field_name)
        if value is None or value == "":
            missing.append(field_name)
        else:
            available.append(field_name)
    return available, missing
