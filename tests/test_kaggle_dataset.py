from pathlib import Path

import pandas as pd

from src.kaggle_dataset import merge_kaggle_csvs


def test_merge_kaggle_csvs(tmp_path):
    input_dir = tmp_path / "kaggle"
    input_dir.mkdir()

    df1 = pd.DataFrame(
        {
            "label": [0, 1],
            "proto": ["tcp", "udp"],
            "value": [1.0, 2.0],
        }
    )
    df2 = pd.DataFrame(
        {
            "label": [0],
            "proto": ["tcp"],
            "value": [3.0],
        }
    )

    df1.to_csv(input_dir / "train.csv", index=False)
    df2.to_csv(input_dir / "test.csv", index=False)

    output_path = tmp_path / "merged.csv"
    merged = merge_kaggle_csvs(input_dir, output_path)

    assert merged == output_path
    merged_df = pd.read_csv(output_path)
    assert len(merged_df) == 3
    assert sorted(merged_df["label"].tolist()) == [0, 0, 1]
