from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _window_label(window_y: np.ndarray, label_strategy: str) -> int:
    if label_strategy == "last":
        return int(window_y[-1])
    if label_strategy == "majority":
        return int(np.sum(window_y) >= (len(window_y) / 2))
    if label_strategy == "any_attack":
        return int(np.any(window_y > 0))
    raise ValueError(f"Unsupported label strategy: {label_strategy}")


@dataclass
class BufferedWindow:
    X_window: np.ndarray
    y_window: int
    start_event_index: int
    end_event_index: int


class RollingWindowBuffer:
    """Collect row-by-row stream events into fixed model windows."""

    def __init__(
        self,
        window_size: int,
        step_size: int,
        label_strategy: str = "any_attack",
    ) -> None:
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive")
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.label_strategy = label_strategy
        self._rows: list[np.ndarray] = []
        self._labels: list[int] = []
        self._event_indices: list[int] = []
        self.events_seen = 0

    def add(self, row, label: int, event_index: int | None = None) -> None:
        row = np.asarray(row, dtype=float)
        if row.ndim != 1:
            raise ValueError("row must be 1D")
        self.events_seen += 1
        self._rows.append(row)
        self._labels.append(int(label))
        self._event_indices.append(self.events_seen if event_index is None else int(event_index))

    def ready(self) -> bool:
        return len(self._rows) >= self.window_size

    def pop_window(self) -> BufferedWindow:
        if not self.ready():
            raise ValueError("buffer does not contain enough rows for a full window")

        X_window = np.stack(self._rows[: self.window_size])
        y_values = np.asarray(self._labels[: self.window_size], dtype=int)
        event_indices = self._event_indices[: self.window_size]
        window = BufferedWindow(
            X_window=X_window,
            y_window=_window_label(y_values, self.label_strategy),
            start_event_index=int(event_indices[0]),
            end_event_index=int(event_indices[-1]),
        )

        self._rows = self._rows[self.step_size :]
        self._labels = self._labels[self.step_size :]
        self._event_indices = self._event_indices[self.step_size :]
        return window

    def pending_count(self) -> int:
        return len(self._rows)
