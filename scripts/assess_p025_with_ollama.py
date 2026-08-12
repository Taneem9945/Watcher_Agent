from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.ollama_client import OllamaClient, OllamaError
from src.mixed_data.llm_assessment import assess_p025_packet
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
    parser = argparse.ArgumentParser(description="Run Ollama assessments over P025 LLM-ready packets.")
    parser.add_argument("--packets", default="results/p025_llm_packets.jsonl")
    parser.add_argument("--output", default="results/p025_ollama_assessments.jsonl")
    parser.add_argument("--summary-output", default="results/p025_ollama_assessments_summary.json")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    packets_path = Path(args.packets)
    if not packets_path.exists():
        raise SystemExit(f"packets file does not exist: {packets_path}")

    packets = _read_jsonl(packets_path)
    start = max(0, args.start_index)
    end = len(packets) if args.max_packets <= 0 else min(len(packets), start + args.max_packets)
    selected_packets = packets[start:end]

    client = OllamaClient(
        base_url=args.base_url,
        model_name=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    output = Path(args.output)
    ensure_dir(output.parent)
    assessments = []
    with output.open("w", encoding="utf-8") as f:
        for index, packet in enumerate(selected_packets, start=start):
            try:
                assessment = assess_p025_packet(client, packet)
            except OllamaError as exc:
                raise SystemExit(f"Ollama failed on packet index {index}: {exc}") from exc
            assessment["packet_index"] = index
            f.write(json.dumps(assessment) + "\n")
            f.flush()
            assessments.append({k: v for k, v in assessment.items() if k not in {"llm_input_packet", "raw_response_text"}})

    summary = {
        "packets_input": str(packets_path),
        "output": str(output),
        "model": args.model,
        "base_url": args.base_url,
        "packet_count_available": len(packets),
        "packet_start_index": start,
        "packet_count_assessed": len(assessments),
        "preview": assessments[:3],
    }
    summary_output = Path(args.summary_output)
    ensure_dir(summary_output.parent)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
