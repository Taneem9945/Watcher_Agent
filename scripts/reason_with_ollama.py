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
from src.stream_memory import StreamMemory
from src.utils import ensure_dir, load_config, split_indices
from src.windowing import create_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--checkpoint", default="checkpoints/best_mamba_watcher.pt")
    parser.add_argument("--output", default="results/ollama_assessments.json")
    parser.add_argument("--raw-output", default="results/ollama_raw_responses.json")
    parser.add_argument("--max-windows", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

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
    X_windows, y_windows = create_windows(
        X,
        y,
        window_size=int(window_cfg["window_size"]),
        step_size=int(window_cfg["step_size"]),
        label_strategy=window_cfg.get("label_strategy", "any_attack"),
    )
    _, _, test_idx = split_indices(
        len(X_windows),
        test_size=float(data_cfg["test_size"]),
        val_size=float(data_cfg["val_size"]),
        seed=int(config["project"]["seed"]),
    )

    model_cfg = config["model"]
    model = MambaWatcher(
        input_dim=len(feature_names),
        d_model=int(model_cfg["d_model"]),
        d_state=int(model_cfg["d_state"]),
        d_conv=int(model_cfg["d_conv"]),
        expand=int(model_cfg["expand"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        pooling=model_cfg.get("pooling", "last"),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(config["project"].get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model.to(device)

    client = OllamaClient(base_url=args.base_url, model_name=args.model)
    watcher = OllamaWatcher(client=client)
    assessments = []
    raw_responses = []
    prev_context_packet = None
    prev_signal_packet = None
    stream_memory = StreamMemory()
    for i, idx in enumerate(test_idx[: args.max_windows]):
        try:
            signal_packet = infer_window_signal(
                model=model,
                X_window=X_windows[idx],
                window_id=int(i),
                true_label=int(y_windows[idx]),
                device=device,
                previous_signal=prev_signal_packet,
            )
            signal_row = signal_packet["signal_rows"][0]
            stream_memory.update(signal_packet)
            context_packet = build_context_packet(
                window_id=int(i),
                X_window=X_windows[idx],
                feature_names=feature_names,
                prediction=int(signal_row["prediction"]),
                attack_probability=float(signal_row["attack_probability"]),
                true_label=int(y_windows[idx]),
                previous_context=prev_context_packet,
                model_signal=signal_row,
                previous_model_signal=(
                    prev_signal_packet["signal_rows"][0] if prev_signal_packet else None
                ),
                stream_memory=stream_memory.snapshot(),
            )
            result = watcher.assess_context_packet(context_packet)
        except OllamaError as exc:
            raise SystemExit(str(exc)) from exc
        result["true_label"] = int(y_windows[idx])
        raw_responses.append(
            {
                "window_id": int(i),
                "mamba_signal_packet": signal_packet,
                "context_packet": context_packet,
                "raw_response_text": result.get("raw_response_text", ""),
                "parsed_response": {k: v for k, v in result.items() if k != "raw_response_text"},
            }
        )
        assessments.append(result)
        prev_context_packet = context_packet
        prev_signal_packet = signal_packet

    ensure_dir("results")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2)
    with open(args.raw_output, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, indent=2)
    print(args.output)
    print(args.raw_output)


if __name__ == "__main__":
    main()
