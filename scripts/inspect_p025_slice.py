from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixed_data.source_manifest import load_p025_manifest, validate_manifest_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the selected P025 prototype data slice.")
    parser.add_argument("--manifest", default="p025_prototype.yaml")
    args = parser.parse_args()

    manifest = load_p025_manifest(args.manifest)
    errors = validate_manifest_files(manifest)
    summary = {
        "name": manifest.name,
        "scenario_root": str(manifest.scenario_root),
        "window_seconds": manifest.window_seconds,
        "selected_sources": [
            {
                "id": source.id,
                "source_type": source.source_type,
                "parser": source.parser,
                "role": source.role,
                "host": source.host,
                "path": str(source.path),
                "resolved_path": str(source.resolved_path),
                "exists": source.resolved_path.exists(),
                "bytes": source.resolved_path.stat().st_size if source.resolved_path.exists() else None,
            }
            for source in manifest.selected_sources
        ],
        "future_sources": list(manifest.future_sources),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
