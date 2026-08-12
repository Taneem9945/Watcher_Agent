from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixed_data.features import windows_to_tensor_batch
from src.mixed_data.parsers import normalize_manifest_events
from src.mixed_data.source_manifest import load_p025_manifest, validate_manifest_files
from src.mixed_data.windowing import create_time_windows
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert P025 mixed event windows into Mamba-ready tensors.")
    parser.add_argument("--manifest", default="p025_prototype.yaml")
    parser.add_argument("--output", default="results/p025_window_tensors.npz")
    parser.add_argument("--summary-output", default="results/p025_window_tensors_summary.json")
    parser.add_argument("--max-events", type=int, default=0, help="Pad/truncate each window to this many events. 0 uses the largest window.")
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()

    manifest = load_p025_manifest(args.manifest)
    errors = validate_manifest_files(manifest)
    if errors:
        raise SystemExit("\n".join(errors))

    events = normalize_manifest_events(manifest)
    windows = create_time_windows(events, window_seconds=manifest.window_seconds)
    batch = windows_to_tensor_batch(
        windows,
        max_events=None if args.max_events <= 0 else args.max_events,
        standardize=not args.no_standardize,
    )

    output = Path(args.output)
    ensure_dir(output.parent)
    np.savez_compressed(
        output,
        X=batch.X,
        mask=batch.mask,
        feature_names=np.asarray(batch.feature_names, dtype=object),
        metadata=np.asarray(batch.metadata_dicts(), dtype=object),
    )

    summary = {
        "output": str(output),
        "standardized": not args.no_standardize,
        **batch.summary(),
    }
    summary_output = Path(args.summary_output)
    ensure_dir(summary_output.parent)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
