from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data_loader import load_network_dataset
from src.mamba_model import MambaWatcher
from src.preprocessing import preprocess_dataframe
from src.train import make_loaders, train_model
from src.utils import load_config, set_seed
from src.windowing import create_windows
from src.utils import split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
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
    if len(X_windows) == 0:
        raise ValueError("No windows were created. Check window_size and input dataset length.")

    train_idx, val_idx, test_idx = split_indices(
        len(X_windows),
        test_size=float(data_cfg["test_size"]),
        val_size=float(data_cfg["val_size"]),
        seed=int(config["project"]["seed"]),
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

    batch_size = int(config["training"]["batch_size"])
    train_loader, val_loader, test_loader = make_loaders(
        X_windows[train_idx],
        y_windows[train_idx],
        X_windows[val_idx],
        y_windows[val_idx],
        X_windows[test_idx],
        y_windows[test_idx],
        batch_size=batch_size,
    )

    result = train_model(model, train_loader, val_loader, config, device)
    history = result["history"]
    if history["val_f1"]:
        best_idx = int(max(range(len(history["val_f1"])), key=lambda i: history["val_f1"][i]))
        print(
            "best_val="
            f"epoch:{best_idx + 1} "
            f"loss:{history['val_loss'][best_idx]:.4f} "
            f"acc:{history['val_accuracy'][best_idx]:.4f} "
            f"prec:{history['val_precision'][best_idx]:.4f} "
            f"rec:{history['val_recall'][best_idx]:.4f} "
            f"f1:{history['val_f1'][best_idx]:.4f} "
            f"fpr:{history['val_fpr'][best_idx]:.4f}"
        )
    print(result["checkpoint_path"])


if __name__ == "__main__":
    main()
