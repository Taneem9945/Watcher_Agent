from pathlib import Path

import yaml

from src.mixed_data.source_manifest import load_p025_manifest, validate_manifest_files


def test_load_p025_manifest_from_file(tmp_path):
    scenario_root = tmp_path / "dataset" / "scenario"
    scenario_root.mkdir(parents=True)
    (scenario_root / "conn.log").write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "prototype": {
                    "name": "test-p025",
                    "dataset_root": str(tmp_path / "dataset"),
                    "scenario_path": "scenario",
                    "window_seconds": 60,
                    "log_year": 2025,
                    "log_timezone": "UTC",
                },
                "selected_sources": [
                    {
                        "id": "zeek_conn",
                        "source_type": "zeek_conn",
                        "parser": "zeek_json",
                        "role": "network_connection",
                        "path": "conn.log",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_p025_manifest(manifest_path)

    assert manifest.name == "test-p025"
    assert manifest.window_seconds == 60
    assert manifest.log_year == 2025
    assert manifest.log_timezone == "UTC"
    assert manifest.scenario_root == Path(tmp_path / "dataset" / "scenario")
    assert manifest.selected_sources[0].resolved_path == scenario_root / "conn.log"
    assert validate_manifest_files(manifest) == []


def test_validate_manifest_files_reports_missing_required_source(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "prototype": {
                    "name": "test-p025",
                    "dataset_root": str(tmp_path / "dataset"),
                    "scenario_path": "scenario",
                    "window_seconds": 60,
                },
                "selected_sources": [
                    {
                        "id": "missing",
                        "source_type": "zeek_http",
                        "parser": "zeek_json",
                        "role": "http_activity",
                        "path": "http.log",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_p025_manifest(manifest_path)
    errors = validate_manifest_files(manifest)

    assert len(errors) == 2
    assert "scenario root does not exist" in errors[0]
    assert "required source does not exist" in errors[1]
