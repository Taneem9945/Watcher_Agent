import numpy as np

from src.windowing import create_windows


def test_window_count_and_labels():
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=int)

    Xw, yw = create_windows(X, y, window_size=4, step_size=2, label_strategy="any_attack")
    assert Xw.shape == (4, 4, 4)
    assert yw.tolist() == [1, 1, 1, 1]

    _, yw_last = create_windows(X, y, window_size=4, step_size=2, label_strategy="last")
    assert yw_last.tolist() == [0, 1, 0, 0]

    _, yw_majority = create_windows(X, y, window_size=4, step_size=2, label_strategy="majority")
    assert yw_majority.tolist() == [0, 1, 1, 0]
