from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from .windowing import iter_window_slices


def build_raw_vs_processed_preview(
    raw_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    window_size: int,
    step_size: int,
    num_rows: int = 5,
    num_windows: int = 3,
):
    """
    Build a JSON-serializable preview of raw rows and transformed model inputs.
    """
    raw_preview = []
    processed_preview = []

    row_limit = min(num_rows, len(raw_df), len(X))
    for i in range(row_limit):
        raw_preview.append(raw_df.iloc[i].to_dict())
        processed_preview.append(
            {
                "row_index": int(i),
                "label": int(y[i]),
                "features": {name: float(value) for name, value in zip(feature_names, X[i])},
            }
        )

    window_preview = []
    for window_id, (start, end) in enumerate(iter_window_slices(len(X), window_size, step_size)):
        if window_id >= num_windows:
            break
        window_preview.append(
            {
                "window_id": int(window_id),
                "row_indices": list(range(start, end)),
                "label": int(y[start:end].max() > 0),
                "raw_rows": raw_df.iloc[start:end].to_dict(orient="records"),
                "processed_window": X[start:end].tolist(),
            }
        )

    return {
        "raw_rows": raw_preview,
        "processed_rows": processed_preview,
        "window_preview": window_preview,
        "feature_names": feature_names,
        "notes": [
            "Raw rows are the original UNSW-NB15 records.",
            "Processed rows are the standardized one-hot encoded features used by the model.",
            "Processed windows are the actual sequence tensors passed to MambaWatcher.",
        ],
    }


def build_side_by_side_frame(
    raw_df: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    row_limit: int = 5,
) -> pd.DataFrame:
    """
    Build a wide comparison frame with raw and processed values side by side.
    """
    row_limit = min(row_limit, len(raw_df), len(X))
    rows: list[dict[str, Any]] = []
    for i in range(row_limit):
        row: dict[str, Any] = {"row_index": int(i), "label": int(y[i])}
        raw_row = raw_df.iloc[i].to_dict()
        for key, value in raw_row.items():
            row[f"raw__{key}"] = value
        for name, value in zip(feature_names, X[i]):
            row[f"s6__{name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_subset_for_window(raw_df: pd.DataFrame, feature_names: list[str], max_features: int = 8):
    numeric_features = [name for name in feature_names if name in raw_df.columns]
    return numeric_features[:max_features]


def save_window_comparison_plot(
    raw_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    window_start: int,
    window_size: int,
    output_path: str | Path,
):
    """
    Save a comparison plot that contrasts raw values and transformed values
    for a small set of numeric features within one window.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_features = _feature_subset_for_window(raw_df, feature_names)
    if not selected_features:
        raise ValueError("No shared numeric features found for plotting")

    raw_window = raw_df.iloc[window_start : window_start + window_size]
    label_window = np.asarray(y[window_start : window_start + window_size], dtype=int)
    transformed_indices = [feature_names.index(name) for name in selected_features]
    transformed_window = X[window_start : window_start + window_size][:, transformed_indices]
    raw_window_values = raw_window[selected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def scale_for_display(values: np.ndarray) -> np.ndarray:
        arr = values.astype(float)
        col_min = np.nanmin(arr, axis=0)
        col_max = np.nanmax(arr, axis=0)
        denom = np.where((col_max - col_min) == 0, 1.0, col_max - col_min)
        return (arr - col_min) / denom

    raw_display = scale_for_display(raw_window_values)
    transformed_display = transformed_window.astype(float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(10, len(selected_features) * 1.2), 9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.35, 1.0, 1.0]},
    )

    label_cmap = ListedColormap(["#2563eb", "#dc2626"])
    label_norm = BoundaryNorm([-0.5, 0.5, 1.5], label_cmap.N)
    label_im = axes[0].imshow(label_window[np.newaxis, :], aspect="auto", interpolation="nearest", cmap=label_cmap, norm=label_norm)
    axes[0].set_title("Row labels across the window")
    axes[0].set_yticks([])
    axes[0].set_xticks(range(window_size))
    axes[0].set_xticklabels([str(i) for i in range(window_start, window_start + window_size)])
    axes[0].set_xlabel("Row index")
    axes[0].text(-0.5, 0, "benign", va="center", ha="right", fontsize=10)
    axes[0].text(-0.5, 1, "attack", va="center", ha="right", fontsize=10)
    cbar = fig.colorbar(label_im, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels(["benign", "attack"])

    raw_im = axes[1].imshow(raw_display.T, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1].set_title("Raw UNSW-NB15 window values, scaled for display")
    axes[1].set_ylabel("Feature")
    axes[1].set_yticks(range(len(selected_features)))
    axes[1].set_yticklabels(selected_features)
    axes[1].set_xticks(range(window_size))
    axes[1].set_xticklabels([str(i) for i in range(window_start, window_start + window_size)])
    fig.colorbar(raw_im, ax=axes[1], fraction=0.046, pad=0.04)

    trans_im = axes[2].imshow(transformed_display.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    axes[2].set_title("Transformed S6/Mamba input window")
    axes[2].set_ylabel("Feature")
    axes[2].set_yticks(range(len(selected_features)))
    axes[2].set_yticklabels(selected_features)
    axes[2].set_xticks(range(window_size))
    axes[2].set_xticklabels([str(i) for i in range(window_start, window_start + window_size)])
    axes[2].set_xlabel("Row index")
    fig.colorbar(trans_im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Raw vs Transformed Window Comparison")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path, selected_features


def build_html_report(
    preview: dict[str, Any],
    side_by_side_frame: pd.DataFrame,
    plot_path: str | Path,
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_path)
    raw_table = pd.DataFrame(preview["raw_rows"]).to_html(index=False, escape=True)
    processed_table = pd.DataFrame(preview["processed_rows"]).to_html(index=False, escape=True)
    window_table = pd.DataFrame(preview["window_preview"]).to_html(index=False, escape=True)
    side_table = side_by_side_frame.to_html(index=False, escape=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>UNSW-NB15 Raw vs S6 Comparison</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1, h2 {{
      color: #0f172a;
    }}
    .card {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      background: white;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
      overflow-x: auto;
      display: block;
      white-space: nowrap;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: left;
    }}
    th {{
      background: #eff6ff;
      position: sticky;
      top: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .meta div {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <h1>UNSW-NB15 Raw vs S6/Mamba Comparison</h1>
  <div class="card meta">
    <div><strong>Preview rows</strong><br>{len(preview["raw_rows"])}</div>
    <div><strong>Preview windows</strong><br>{len(preview["window_preview"])}</div>
    <div><strong>Features</strong><br>{len(preview["feature_names"])}</div>
    <div><strong>Plot</strong><br>{plot_path.name}</div>
  </div>
  <div class="card">
    <h2>Window Plot</h2>
    <img src="{plot_path.name}" alt="Raw vs transformed window plot" />
  </div>
  <div class="card">
    <h2>Side-by-Side CSV View</h2>
    {side_table}
  </div>
  <div class="card">
    <h2>Raw Rows</h2>
    {raw_table}
  </div>
  <div class="card">
    <h2>Processed Rows</h2>
    {processed_table}
  </div>
  <div class="card">
    <h2>Window Preview</h2>
    {window_table}
  </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
