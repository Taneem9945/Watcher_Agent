from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixed_data.llm_packets import assert_no_forbidden_keys, build_p025_llm_packet
from src.utils import ensure_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM-ready packets for P025 mixed-data windows.")
    parser.add_argument("--windows", default="results/p025_windows.jsonl")
    parser.add_argument("--encoder-signals", default="results/p025_mamba_encoder_signals.jsonl")
    parser.add_argument("--output", default="results/p025_llm_packets.jsonl")
    parser.add_argument("--preview-output", default="results/p025_llm_packets_preview.json")
    parser.add_argument("--evidence-limit", type=int, default=12)
    parser.add_argument("--preview", type=int, default=3)
    args = parser.parse_args()

    windows_path = Path(args.windows)
    signals_path = Path(args.encoder_signals)
    if not windows_path.exists():
        raise SystemExit(f"windows file does not exist: {windows_path}")
    if not signals_path.exists():
        raise SystemExit(f"encoder signals file does not exist: {signals_path}")

    windows = _read_jsonl(windows_path)
    encoder_signals = _read_jsonl(signals_path)
    signals_by_window_id = {int(signal["window_id"]): signal for signal in encoder_signals}

    packets = []
    previous_signals: list[dict[str, Any]] = []
    for window in windows:
        window_id = int(window["window_id"])
        encoder_signal = signals_by_window_id.get(window_id)
        if encoder_signal is None:
            raise SystemExit(f"missing encoder signal for window_id={window_id}")
        packet = build_p025_llm_packet(
            window=window,
            encoder_signal=encoder_signal,
            previous_encoder_signals=previous_signals,
            evidence_limit=args.evidence_limit,
        )
        assert_no_forbidden_keys(packet)
        packets.append(packet)
        previous_signals.append(encoder_signal)

    output = Path(args.output)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        for packet in packets:
            f.write(json.dumps(packet) + "\n")

    preview = {
        "output": str(output),
        "packet_count": len(packets),
        "windows_input": str(windows_path),
        "encoder_signals_input": str(signals_path),
        "preview": packets[: max(0, args.preview)],
    }
    preview_output = Path(args.preview_output)
    ensure_dir(preview_output.parent)
    with preview_output.open("w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2)
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
