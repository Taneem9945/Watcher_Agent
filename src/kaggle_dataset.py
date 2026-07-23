from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def discover_csv_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() == ".csv":
        return [root_path]
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    return sorted(p for p in root_path.rglob("*.csv") if p.is_file())


def merge_kaggle_csvs(
    input_root: str | Path,
    output_path: str | Path,
    required_columns: Iterable[str] = ("label",),
) -> Path:
    """
    Merge Kaggle-exported CSV files into a single raw dataset file.

    The UNSW-NB15 Kaggle dataset is often split across train/test CSVs. This helper
    selects CSVs that contain the required columns and concatenates them into one file.
    """
    csv_files = discover_csv_files(input_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_root}")

    required_columns = list(required_columns)
    candidate_files = []
    for csv_file in csv_files:
        try:
            header = _read_csv_header(csv_file)
        except Exception:
            continue
        if all(col in header for col in required_columns):
            candidate_files.append(csv_file)

    if not candidate_files:
        candidate_files = csv_files

    frames = [pd.read_csv(csv_file) for csv_file in candidate_files]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path
