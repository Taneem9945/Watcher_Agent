from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def preprocess_dataframe(
    df,
    label_col: str,
    categorical_columns: list[str],
    drop_columns: list[str],
    binary_label: bool = True,
    attack_label_values: list | None = None,
):
    """
    Convert a raw network-flow dataframe into numeric features and labels.

    Requirements:
    - Drop configured columns.
    - Separate features and labels.
    - Convert labels to binary if binary_label=True.
    - One-hot encode categorical columns.
    - Fill missing numeric values with median.
    - Standardize numeric features.
    - Return X, y, feature_names, preprocessing_artifacts.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' is missing from the dataframe")

    attack_label_values = attack_label_values or []
    attack_labels = {_normalize_label(v) for v in attack_label_values}

    labels = df[label_col].copy()
    if binary_label:
        y = labels.map(lambda v: 1 if _normalize_label(v) in attack_labels else 0).to_numpy(dtype=np.int64)
    else:
        codes, uniques = pd.factorize(labels)
        y = codes.astype(np.int64)

    feature_df = df.drop(columns=[label_col], errors="ignore")
    feature_df = feature_df.drop(columns=[c for c in drop_columns if c in feature_df.columns], errors="ignore")

    cat_cols = [c for c in categorical_columns if c in feature_df.columns]
    numeric_df = feature_df.drop(columns=cat_cols, errors="ignore").copy()

    if not numeric_df.empty:
        numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
        medians = numeric_df.median(numeric_only=True)
        numeric_df = numeric_df.fillna(medians)
    else:
        medians = pd.Series(dtype=float)

    if cat_cols:
        categorical_df = feature_df[cat_cols].copy().fillna("missing").astype(str)
        categorical_df = pd.get_dummies(categorical_df, columns=cat_cols, dtype=float)
    else:
        categorical_df = pd.DataFrame(index=feature_df.index)

    processed_df = pd.concat([numeric_df, categorical_df], axis=1)
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = processed_df.columns.tolist()
    processed_df = processed_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(processed_df.to_numpy(dtype=float))
    feature_names = numeric_cols

    preprocessing_artifacts = {
        "scaler": scaler,
        "numeric_medians": medians.to_dict(),
        "categorical_columns": cat_cols,
        "drop_columns": [c for c in drop_columns if c in df.columns],
        "feature_names": feature_names,
        "binary_label": binary_label,
        "attack_label_values": attack_label_values,
    }
    return X.astype(np.float32), y, feature_names, preprocessing_artifacts
