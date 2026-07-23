from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | os.PathLike[str], data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def split_indices(
    n_items: int,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    indices = np.arange(n_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_count = max(1, int(round(n_items * test_size)))
    val_count = max(1, int(round(n_items * val_size)))
    train_count = n_items - test_count - val_count
    if train_count <= 0:
        raise ValueError("Not enough items to create train/val/test splits")

    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]
    return train_idx, val_idx, test_idx


def to_device(batch, device: torch.device | str):
    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(item, device) for item in batch)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch
