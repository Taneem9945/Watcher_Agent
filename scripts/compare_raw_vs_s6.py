from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_network_dataset
from src.inspection import (
    build_html_report,
    build_raw_vs_processed_preview,
    build_side_by_side_frame,
    save_window_comparison_plot,
)
from src.preprocessing import preprocess_dataframe
from src.utils import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--output", default="results/raw_vs_s6_preview.json")
    parser.add_argument("--csv-output", default="results/raw_vs_s6_side_by_side.csv")
    parser.add_argument("--html-output", default="results/raw_vs_s6_report.html")
    parser.add_argument("--plot-output", default="results/raw_vs_s6_window_plot.png")
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

    preview = build_raw_vs_processed_preview(
        raw_df=df,
        X=X,
        y=y,
        feature_names=feature_names,
        window_size=int(config["windowing"]["window_size"]),
        step_size=int(config["windowing"]["step_size"]),
        num_rows=args.rows,
        num_windows=args.windows,
    )

    side_by_side = build_side_by_side_frame(
        raw_df=df,
        X=X,
        feature_names=feature_names,
        y=y,
        row_limit=args.rows,
    )
    side_by_side.to_csv(args.csv_output, index=False)

    window_size = int(config["windowing"]["window_size"])
    plot_path, _ = save_window_comparison_plot(
        raw_df=df,
        X=X,
        y=y,
        feature_names=feature_names,
        window_start=0,
        window_size=window_size,
        output_path=args.plot_output,
    )

    ensure_dir("results")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2)
    build_html_report(
        preview=preview,
        side_by_side_frame=side_by_side,
        plot_path=plot_path,
        output_path=args.html_output,
    )
    print(args.output)
    print(args.csv_output)
    print(args.html_output)
    print(args.plot_output)


if __name__ == "__main__":
    main()
