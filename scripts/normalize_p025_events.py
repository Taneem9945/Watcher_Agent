from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixed_data.parsers import normalize_manifest_events
from src.mixed_data.source_manifest import load_p025_manifest, validate_manifest_files
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the selected P025 prototype sources into one event stream.")
    parser.add_argument("--manifest", default="p025_prototype.yaml")
    parser.add_argument("--output", default="results/p025_normalized_events.jsonl")
    parser.add_argument("--preview", type=int, default=5)
    args = parser.parse_args()

    manifest = load_p025_manifest(args.manifest)
    errors = validate_manifest_files(manifest)
    if errors:
        raise SystemExit("\n".join(errors))

    events = normalize_manifest_events(manifest)
    output = Path(args.output)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.to_dict()) + "\n")

    counts = Counter(event.source_type for event in events)
    summary = {
        "output": str(output),
        "event_count": len(events),
        "source_counts": dict(sorted(counts.items())),
        "first_timestamp": events[0].timestamp if events else None,
        "last_timestamp": events[-1].timestamp if events else None,
        "preview": [event.to_dict() for event in events[: max(0, args.preview)]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
