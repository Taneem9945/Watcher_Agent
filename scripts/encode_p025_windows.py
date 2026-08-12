from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mamba_model import MambaWatcher
from src.mamba_signal import infer_window_encoder_signal
from src.utils import ensure_dir


def _load_npz_object_array(value: np.ndarray) -> list[Any]:
    return [item.item() if hasattr(item, "item") else item for item in value]


def _load_model(
    *,
    input_dim: int,
    checkpoint_path: str | None,
    device: torch.device,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    dropout: float,
    pooling: str,
) -> tuple[MambaWatcher, bool]:
    model = MambaWatcher(
        input_dim=input_dim,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_classes=2,
        dropout=dropout,
        pooling=pooling,
    )
    loaded_checkpoint = False
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        loaded_checkpoint = True
    model.to(device)
    return model, loaded_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P025 mixed-window tensors through the Mamba encoder.")
    parser.add_argument("--tensor-input", default="results/p025_window_tensors.npz")
    parser.add_argument("--output", default="results/p025_mamba_encoder_signals.jsonl")
    parser.add_argument("--summary-output", default="results/p025_mamba_encoder_summary.json")
    parser.add_argument("--checkpoint", default=None, help="Optional compatible MambaWatcher checkpoint.")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling", choices=["last", "mean"], default="mean")
    args = parser.parse_args()

    tensor_path = Path(args.tensor_input)
    if not tensor_path.exists():
        raise SystemExit(f"tensor input does not exist: {tensor_path}")

    artifact = np.load(tensor_path, allow_pickle=True)
    X = artifact["X"].astype(np.float32)
    mask = artifact["mask"].astype(np.float32)
    feature_names = [str(item) for item in artifact["feature_names"].tolist()]
    metadata = _load_npz_object_array(artifact["metadata"])

    if X.ndim != 3:
        raise SystemExit(f"expected X to have shape [windows, events, features], got {X.shape}")
    if mask.shape != X.shape[:2]:
        raise SystemExit(f"mask shape {mask.shape} does not match X window/event shape {X.shape[:2]}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, loaded_checkpoint = _load_model(
        input_dim=X.shape[-1],
        checkpoint_path=args.checkpoint,
        device=device,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        pooling=args.pooling,
    )

    limit = X.shape[0] if args.max_windows <= 0 else min(args.max_windows, X.shape[0])
    output = Path(args.output)
    ensure_dir(output.parent)

    records = []
    previous_signal = None
    with output.open("w", encoding="utf-8") as f:
        for i in range(limit):
            window_metadata = dict(metadata[i])
            window_id = int(window_metadata.get("window_id", i))
            packet = infer_window_encoder_signal(
                model=model,
                X_window=X[i],
                window_id=window_id,
                mask=mask[i],
                device=device,
                metadata=window_metadata,
                previous_signal=previous_signal,
            )
            signal_row = packet["signal_rows"][0]
            f.write(json.dumps(signal_row) + "\n")
            records.append(signal_row)
            previous_signal = packet

    summary = {
        "tensor_input": str(tensor_path),
        "output": str(output),
        "loaded_checkpoint": loaded_checkpoint,
        "note": "Without a compatible checkpoint, this is a plumbing/shape-valid encoder pass, not a trained security representation.",
        "input_shape": list(X.shape),
        "mask_shape": list(mask.shape),
        "feature_count": len(feature_names),
        "encoded_windows": len(records),
        "pooling": args.pooling,
        "d_model": args.d_model,
        "preview": records[:3],
    }
    summary_output = Path(args.summary_output)
    ensure_dir(summary_output.parent)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
