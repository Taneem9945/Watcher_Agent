from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixed_data.parsers import normalize_manifest_events
from src.mixed_data.source_manifest import load_p025_manifest, validate_manifest_files
from src.mixed_data.windowing import create_time_windows
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed-time windows from the normalized P025 event stream.")
    parser.add_argument("--manifest", default="p025_prototype.yaml")
    parser.add_argument("--output", default="results/p025_windows.jsonl")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--preview", type=int, default=3)
    args = parser.parse_args()

    manifest = load_p025_manifest(args.manifest)
    errors = validate_manifest_files(manifest)
    if errors:
        raise SystemExit("\n".join(errors))

    events = normalize_manifest_events(manifest)
    windows = create_time_windows(
        events,
        window_seconds=manifest.window_seconds,
        include_empty=args.include_empty,
    )

    output = Path(args.output)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        for window in windows:
            f.write(json.dumps(window.to_dict()) + "\n")

    summary = {
        "output": str(output),
        "window_seconds": manifest.window_seconds,
        "include_empty": bool(args.include_empty),
        "source_event_count": len(events),
        "window_count": len(windows),
        "first_window": windows[0].start_timestamp if windows else None,
        "last_window": windows[-1].start_timestamp if windows else None,
        "preview": [window.to_dict() for window in windows[: max(0, args.preview)]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
