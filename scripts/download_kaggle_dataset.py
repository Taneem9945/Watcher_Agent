from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kaggle_dataset import merge_kaggle_csvs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="mrwellsdavid/unsw-nb15",
        help="Kaggle dataset slug for kagglehub.dataset_download",
    )
    parser.add_argument(
        "--output",
        default="data/raw/network_data.csv",
        help="Where to write the merged CSV",
    )
    parser.add_argument(
        "--required-column",
        action="append",
        default=None,
        help="Columns that must exist for a CSV to be included",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Add it to your environment first."
        ) from exc

    download_path = kagglehub.dataset_download(args.dataset)
    required_columns = args.required_column or ["label"]
    merged = merge_kaggle_csvs(
        download_path,
        args.output,
        required_columns=required_columns,
    )
    print(str(merged))


if __name__ == "__main__":
    main()
