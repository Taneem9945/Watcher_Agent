from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceSpec:
    id: str
    source_type: str
    parser: str
    role: str
    path: Path
    resolved_path: Path
    required: bool = True
    host: str | None = None


@dataclass(frozen=True)
class P025PrototypeManifest:
    name: str
    description: str
    dataset_root: Path
    scenario_path: Path
    scenario_root: Path
    window_seconds: int
    log_year: int
    log_timezone: str
    selected_sources: tuple[SourceSpec, ...]
    future_sources: tuple[str, ...]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _require_mapping(data, "manifest")


def load_p025_manifest(path: str | Path = "p025_prototype.yaml") -> P025PrototypeManifest:
    manifest_path = Path(path)
    data = _load_yaml(manifest_path)
    prototype = _require_mapping(data.get("prototype"), "prototype")
    selected = data.get("selected_sources")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected_sources must be a non-empty list")

    dataset_root = Path(prototype["dataset_root"])
    scenario_path = Path(prototype["scenario_path"])
    scenario_root = dataset_root / scenario_path
    window_seconds = int(prototype.get("window_seconds", 60))
    if window_seconds <= 0:
        raise ValueError("prototype.window_seconds must be positive")
    log_year = int(prototype.get("log_year", 2025))
    log_timezone = str(prototype.get("log_timezone", "UTC"))

    source_specs: list[SourceSpec] = []
    for index, item in enumerate(selected):
        source = _require_mapping(item, f"selected_sources[{index}]")
        rel_path = Path(source["path"])
        source_specs.append(
            SourceSpec(
                id=str(source["id"]),
                source_type=str(source["source_type"]),
                parser=str(source["parser"]),
                role=str(source["role"]),
                path=rel_path,
                resolved_path=scenario_root / rel_path,
                required=bool(source.get("required", True)),
                host=None if source.get("host") is None else str(source["host"]),
            )
        )

    return P025PrototypeManifest(
        name=str(prototype["name"]),
        description=str(prototype.get("description", "")),
        dataset_root=dataset_root,
        scenario_path=scenario_path,
        scenario_root=scenario_root,
        window_seconds=window_seconds,
        log_year=log_year,
        log_timezone=log_timezone,
        selected_sources=tuple(source_specs),
        future_sources=tuple(str(item) for item in data.get("future_sources", [])),
    )


def validate_manifest_files(manifest: P025PrototypeManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.scenario_root.exists():
        errors.append(f"scenario root does not exist: {manifest.scenario_root}")

    for source in manifest.selected_sources:
        if source.required and not source.resolved_path.exists():
            errors.append(f"required source does not exist: {source.id} -> {source.resolved_path}")
        elif source.resolved_path.exists() and not source.resolved_path.is_file():
            errors.append(f"source is not a file: {source.id} -> {source.resolved_path}")
    return errors
