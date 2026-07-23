from __future__ import annotations

import numpy as np


def _window_label(window_y: np.ndarray, label_strategy: str) -> int:
    if label_strategy == "last":
        return int(window_y[-1])
    if label_strategy == "majority":
        return int(np.sum(window_y) >= (len(window_y) / 2))
    if label_strategy == "any_attack":
        return int(np.any(window_y > 0))
    raise ValueError(f"Unsupported label strategy: {label_strategy}")


def iter_window_slices(num_rows: int, window_size: int, step_size: int):
    """
    Yield (start, end) index pairs for each window.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    if num_rows < window_size:
        return
    for start in range(0, num_rows - window_size + 1, step_size):
        yield start, start + window_size


def create_windows(
    X,
    y,
    window_size: int,
    step_size: int,
    label_strategy: str = "any_attack",
):
    """
    Create fixed-length sequence windows.

    Supported label strategies:
    - "last": use label of final row in the window
    - "majority": use majority label inside the window
    - "any_attack": label window as attack if any row in the window is attack

    Default should be "any_attack".
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape [num_flows, feature_dim]")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of shape [num_flows]")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    if len(X) < window_size:
        return np.empty((0, window_size, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)

    windows = []
    labels = []
    for start, end in iter_window_slices(len(X), window_size, step_size):
        end = start + window_size
        x_window = X[start:end]
        y_window = y[start:end]
        windows.append(x_window)
        labels.append(_window_label(y_window, label_strategy))

    return np.stack(windows), np.asarray(labels, dtype=y.dtype)
