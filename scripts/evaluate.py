from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.data_loader import load_network_dataset
from src.evaluate import evaluate_model
from src.mamba_model import MambaWatcher
from src.preprocessing import preprocess_dataframe
from src.train import make_loaders
from src.utils import ensure_dir, load_config, split_indices
from src.windowing import create_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_mamba_watcher.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = config["project"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    data_cfg = config["data"]
    df = load_network_dataset(
        data_cfg["raw_path"],
        label_col=data_cfg["label_col"],
        timestamp_col=data_cfg.get("timestamp_col"),
    )
    X, y, feature_names, _ = preprocess_dataframe(
        df,
        label_col=data_cfg["label_col"],
        categorical_columns=data_cfg.get("categorical_columns", []),
        drop_columns=list(data_cfg.get("drop_columns", []))
        + ([data_cfg["timestamp_col"]] if data_cfg.get("timestamp_col") in df.columns else []),
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
    train_idx, val_idx, test_idx = split_indices(
        len(X_windows),
        test_size=float(data_cfg["test_size"]),
        val_size=float(data_cfg["val_size"]),
        seed=int(config["project"]["seed"]),
    )
    _, _, test_loader = make_loaders(
        X_windows[train_idx],
        y_windows[train_idx],
        X_windows[val_idx],
        y_windows[val_idx],
        X_windows[test_idx],
        y_windows[test_idx],
        batch_size=int(config["training"]["batch_size"]),
    )

    model_cfg = config["model"]
    model_cfg["input_dim"] = len(feature_names)
    model = MambaWatcher(
        input_dim=model_cfg["input_dim"],
        d_model=int(model_cfg["d_model"]),
        d_state=int(model_cfg["d_state"]),
        d_conv=int(model_cfg["d_conv"]),
        expand=int(model_cfg["expand"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        pooling=model_cfg.get("pooling", "last"),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    report = evaluate_model(model, test_loader, device)
    ensure_dir("results")
    with open("results/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("results/evaluation_report.json")


if __name__ == "__main__":
    main()
