from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .events import NormalizedEvent, compute_available_missing
from .source_manifest import P025PrototypeManifest, SourceSpec


AUTH_LINE_RE = re.compile(r"^(?P<month>\w{3})\s+(?P<day>\d{1,2}) (?P<time>\d\d:\d\d:\d\d) (?P<host>\S+) (?P<message>.+)$")
USERNAME_PATTERNS = (
    re.compile(r"\bname=(?P<user>[A-Za-z_][A-Za-z0-9_.-]*)"),
    re.compile(r"\bUSER=(?P<user>[A-Za-z_][A-Za-z0-9_.-]*)"),
    re.compile(r"\bfor user (?P<user>[A-Za-z_][A-Za-z0-9_.-]*)"),
)


def _event_id(source: SourceSpec, timestamp_epoch: float, line_number: int, summary: str) -> str:
    digest = hashlib.sha1(f"{source.id}|{timestamp_epoch}|{line_number}|{summary}".encode("utf-8")).hexdigest()[:12]
    return f"{source.id}:{line_number}:{digest}"


def _iso_from_epoch(timestamp_epoch: float) -> str:
    return datetime.fromtimestamp(float(timestamp_epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_auth_timestamp(line: str, log_year: int) -> tuple[datetime, str, str] | None:
    match = AUTH_LINE_RE.match(line)
    if not match:
        return None
    timestamp = datetime.strptime(
        f"{log_year} {match.group('month')} {match.group('day')} {match.group('time')}",
        "%Y %b %d %H:%M:%S",
    ).replace(tzinfo=timezone.utc)
    return timestamp, match.group("host"), match.group("message")


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _auth_event_type(message: str) -> str:
    lowered = message.lower()
    if "new user:" in lowered:
        return "user_created"
    if "new group:" in lowered:
        return "group_created"
    if "password changed" in lowered:
        return "password_changed"
    if "sudo:" in lowered and "command=" in lowered:
        return "sudo_command"
    if "session opened" in lowered:
        return "session_opened"
    if "session closed" in lowered:
        return "session_closed"
    if "failed password" in lowered:
        return "failed_login"
    if "accepted password" in lowered or "accepted publickey" in lowered:
        return "successful_login"
    return "auth_event"


def _auth_features(message: str) -> dict[str, int]:
    lowered = message.lower()
    return {
        "event_user_created": int("new user:" in lowered),
        "event_group_created": int("new group:" in lowered),
        "event_password_changed": int("password changed" in lowered),
        "event_sudo_used": int("sudo:" in lowered),
        "event_ssh_keygen": int("ssh-keygen" in lowered),
        "event_session_opened": int("session opened" in lowered),
        "event_session_closed": int("session closed" in lowered),
        "command_present": int("command=" in lowered),
    }


def _extract_username(message: str) -> str | None:
    for pattern in USERNAME_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group("user")
    return None


def _http_features(record: dict[str, Any]) -> dict[str, Any]:
    uri = str(record.get("uri") or "")
    user_agent = str(record.get("user_agent") or "")
    lowered_uri = uri.lower()
    lowered_agent = user_agent.lower()
    return {
        "uri_present": int(bool(uri)),
        "uri_length": len(uri),
        "has_query_string": int("?" in uri),
        "has_cmd_param": int("cmd=" in lowered_uri),
        "has_passwd_keyword": int("passwd" in lowered_uri),
        "has_shadow_keyword": int("shadow" in lowered_uri),
        "path_depth": uri.count("/"),
        "file_extension_jsp": int(".jsp" in lowered_uri),
        "file_extension_py": int(".py" in lowered_uri),
        "user_agent_wget": int("wget" in lowered_agent),
        "user_agent_python_requests": int("python-requests" in lowered_agent),
        "user_agent_cli_tool": int(("wget" in lowered_agent) or ("python-requests" in lowered_agent)),
        "request_body_len": _int_or_none(record.get("request_body_len")),
        "response_body_len": _int_or_none(record.get("response_body_len")),
        "status_code": _int_or_none(record.get("status_code")),
    }


def _conn_features(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration": record.get("duration"),
        "orig_bytes": record.get("orig_bytes"),
        "resp_bytes": record.get("resp_bytes"),
        "orig_pkts": record.get("orig_pkts"),
        "resp_pkts": record.get("resp_pkts"),
        "missed_bytes": record.get("missed_bytes"),
        "src_ip_present": int(bool(record.get("id.orig_h"))),
        "dst_ip_present": int(bool(record.get("id.resp_h"))),
    }


def _zeek_event_type(source: SourceSpec, record: dict[str, Any]) -> str:
    if source.source_type == "zeek_conn":
        service = record.get("service")
        return f"connection_{service}" if service else "connection"
    if source.source_type == "zeek_http":
        return "http_request"
    return source.source_type


def _zeek_summary(source: SourceSpec, record: dict[str, Any]) -> str:
    src = record.get("id.orig_h", "unknown")
    dst = record.get("id.resp_h", "unknown")
    dst_port = record.get("id.resp_p", "unknown")
    if source.source_type == "zeek_http":
        method = record.get("method", "HTTP")
        host = record.get("host", "")
        uri = record.get("uri", "")
        return f"{method} {host}{uri} from {src} to {dst}:{dst_port}"
    proto = record.get("proto", "unknown")
    service = record.get("service", "unknown")
    return f"{proto}/{service} connection from {src} to {dst}:{dst_port}"


def parse_zeek_json(source: SourceSpec, _: P025PrototypeManifest) -> Iterable[NormalizedEvent]:
    with source.resolved_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            timestamp_epoch = float(record["ts"])
            summary = _zeek_summary(source, record)
            values = {
                "src_ip": record.get("id.orig_h"),
                "dst_ip": record.get("id.resp_h"),
                "src_port": record.get("id.orig_p"),
                "dst_port": record.get("id.resp_p"),
                "username": None,
                "uid": record.get("uid"),
            }
            available, missing = compute_available_missing(
                values,
                ["src_ip", "dst_ip", "src_port", "dst_port", "username", "uid"],
            )
            features = _http_features(record) if source.source_type == "zeek_http" else _conn_features(record)
            yield NormalizedEvent(
                event_id=_event_id(source, timestamp_epoch, line_number, summary),
                timestamp=_iso_from_epoch(timestamp_epoch),
                timestamp_epoch=timestamp_epoch,
                source_id=source.id,
                source_type=source.source_type,
                parser=source.parser,
                role=source.role,
                host=source.host,
                src_ip=_first_string(record.get("id.orig_h")),
                dst_ip=_first_string(record.get("id.resp_h")),
                src_port=_int_or_none(record.get("id.orig_p")),
                dst_port=_int_or_none(record.get("id.resp_p")),
                username=None,
                uid=_first_string(record.get("uid")),
                event_type=_zeek_event_type(source, record),
                summary=summary,
                features=features,
                stable_ids={
                    "uid": record.get("uid"),
                    "community_id": record.get("community_id"),
                    "src_ip": record.get("id.orig_h"),
                    "dst_ip": record.get("id.resp_h"),
                },
                available_fields=available,
                missing_fields=missing,
            )


def parse_linux_auth_text(source: SourceSpec, manifest: P025PrototypeManifest) -> Iterable[NormalizedEvent]:
    with source.resolved_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            parsed = _parse_auth_timestamp(line, manifest.log_year)
            if parsed is None:
                continue
            timestamp, host, message = parsed
            username = _extract_username(message)
            timestamp_epoch = timestamp.timestamp()
            event_type = _auth_event_type(message)
            values = {
                "src_ip": None,
                "dst_ip": None,
                "src_port": None,
                "dst_port": None,
                "username": username,
                "uid": None,
            }
            available, missing = compute_available_missing(
                values,
                ["src_ip", "dst_ip", "src_port", "dst_port", "username", "uid"],
            )
            yield NormalizedEvent(
                event_id=_event_id(source, timestamp_epoch, line_number, message),
                timestamp=timestamp.isoformat().replace("+00:00", "Z"),
                timestamp_epoch=timestamp_epoch,
                source_id=source.id,
                source_type=source.source_type,
                parser=source.parser,
                role=source.role,
                host=host,
                src_ip=None,
                dst_ip=None,
                src_port=None,
                dst_port=None,
                username=username,
                uid=None,
                event_type=event_type,
                summary=f"{host}: {message}",
                features={
                    **_auth_features(message),
                    "username_present": int(username is not None),
                    "src_ip_present": 0,
                    "dst_ip_present": 0,
                },
                stable_ids={
                    "host": host,
                    "username": username,
                },
                available_fields=available,
                missing_fields=missing,
            )


PARSER_REGISTRY = {
    "zeek_json": parse_zeek_json,
    "linux_auth_text": parse_linux_auth_text,
}


def normalize_manifest_events(manifest: P025PrototypeManifest) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []
    for source in manifest.selected_sources:
        parser = PARSER_REGISTRY.get(source.parser)
        if parser is None:
            raise ValueError(f"unsupported parser for {source.id}: {source.parser}")
        events.extend(parser(source, manifest))
    return sorted(events, key=lambda event: (event.timestamp_epoch, event.source_id, event.event_id))
