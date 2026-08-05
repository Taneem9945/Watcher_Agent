from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.context_builder import build_context_packet
from src.data_loader import load_network_dataset
from src.llm.ollama_client import OllamaClient, OllamaError
from src.llm.ollama_watcher import OllamaWatcher
from src.mamba_model import MambaWatcher
from src.mamba_signal import infer_window_signal
from src.preprocessing import preprocess_dataframe
from src.stream_buffer import RollingWindowBuffer
from src.stream_memory import StreamMemory
from src.utils import ensure_dir, load_config


def _load_model(config: dict, checkpoint_path: str, device: torch.device, input_dim: int) -> MambaWatcher:
    model_cfg = config["model"]
    model = MambaWatcher(
        input_dim=input_dim,
        d_model=int(model_cfg["d_model"]),
        d_state=int(model_cfg["d_state"]),
        d_conv=int(model_cfg["d_conv"]),
        expand=int(model_cfg["expand"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        pooling=model_cfg.get("pooling", "last"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_mamba_watcher.pt")
    parser.add_argument("--output", default="results/live_replay.jsonl")
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    device_name = config["project"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    df = load_network_dataset(
        data_cfg["raw_path"],
        label_col=data_cfg["label_col"],
        timestamp_col=data_cfg.get("timestamp_col"),
    )
    drop_columns = list(data_cfg.get("drop_columns", []))
    if data_cfg.get("timestamp_col") and data_cfg["timestamp_col"] in df.columns:
        drop_columns.append(data_cfg["timestamp_col"])

    X, y, feature_names, _ = preprocess_dataframe(
        df,
        label_col=data_cfg["label_col"],
        categorical_columns=data_cfg.get("categorical_columns", []),
        drop_columns=drop_columns,
        binary_label=bool(data_cfg.get("binary_label", True)),
        attack_label_values=data_cfg.get("attack_label_values", []),
    )

    window_cfg = config["windowing"]
    buffer = RollingWindowBuffer(
        window_size=int(window_cfg["window_size"]),
        step_size=int(window_cfg["step_size"]),
        label_strategy=window_cfg.get("label_strategy", "any_attack"),
    )
    model = _load_model(config, args.checkpoint, device, len(feature_names))
    client = OllamaClient(base_url=args.base_url, model_name=args.model)
    watcher = OllamaWatcher(client=client)

    stream_memory = StreamMemory()
    prev_context_packet = None
    prev_signal_packet = None
    windows_written = 0

    ensure_dir(Path(args.output).parent)
    with open(args.output, "w", encoding="utf-8") as out:
        for row_index in range(max(0, args.start_row), len(X)):
            buffer.add(X[row_index], int(y[row_index]), event_index=row_index)
            while buffer.ready() and windows_written < args.max_windows:
                buffered = buffer.pop_window()
                signal_packet = infer_window_signal(
                    model=model,
                    X_window=buffered.X_window,
                    window_id=windows_written,
                    true_label=buffered.y_window,
                    device=device,
                    previous_signal=prev_signal_packet,
                )
                signal_row = signal_packet["signal_rows"][0]
                stream_memory_before = stream_memory.snapshot()
                stream_memory_after = stream_memory.update(signal_packet)

                context_packet = build_context_packet(
                    window_id=windows_written,
                    X_window=buffered.X_window,
                    feature_names=feature_names,
                    prediction=int(signal_row["prediction"]),
                    attack_probability=float(signal_row["attack_probability"]),
                    true_label=buffered.y_window,
                    previous_context=prev_context_packet,
                    model_signal=signal_row,
                    previous_model_signal=(prev_signal_packet["signal_rows"][0] if prev_signal_packet else None),
                    stream_memory=stream_memory_after,
                )

                record = {
                    "window_id": windows_written,
                    "events_seen": int(buffer.events_seen),
                    "source_row_start": buffered.start_event_index,
                    "source_row_end": buffered.end_event_index,
                    "pending_buffer_rows": buffer.pending_count(),
                    "true_label": buffered.y_window,
                    "mamba_signal_packet": signal_packet,
                    "stream_memory_before": stream_memory_before,
                    "stream_memory_after": stream_memory_after,
                    "context_packet": context_packet,
                }

                try:
                    result = watcher.assess_context_packet(context_packet, prompt_mode="encoder")
                except OllamaError as exc:
                    raise SystemExit(str(exc)) from exc
                record["llm_assessment"] = {k: v for k, v in result.items() if k != "raw_response_text"}
                record["raw_response_text"] = result.get("raw_response_text", "")

                out.write(json.dumps(record) + "\n")
                prev_context_packet = context_packet
                prev_signal_packet = signal_packet
                windows_written += 1

            if windows_written >= args.max_windows:
                break

    print(args.output)


if __name__ == "__main__":
    main()
