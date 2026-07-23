from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_network_dataset(path: str, label_col: str, timestamp_col: str | None = None):
    """
    Load a CSV network-flow dataset.

    Requirements:
    - Load CSV with pandas.
    - Validate that label_col exists.
    - If timestamp_col exists, sort by timestamp.
    - If timestamp_col is None or missing, preserve row order.
    - Return pandas DataFrame.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")

    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(by=timestamp_col, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df
