import numpy as np

from src.stream_buffer import RollingWindowBuffer


def test_rolling_window_buffer_slides_by_step_size():
    buffer = RollingWindowBuffer(window_size=3, step_size=2, label_strategy="any_attack")

    for i in range(5):
        buffer.add(np.array([i, i + 1]), label=1 if i == 2 else 0, event_index=i)

    first = buffer.pop_window()
    assert first.X_window.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert first.y_window == 1
    assert first.start_event_index == 0
    assert first.end_event_index == 2
    assert buffer.pending_count() == 3

    second = buffer.pop_window()
    assert second.X_window.tolist() == [[2, 3], [3, 4], [4, 5]]
    assert second.y_window == 1
    assert second.start_event_index == 2
    assert second.end_event_index == 4
    assert buffer.pending_count() == 1


def test_rolling_window_buffer_majority_label():
    buffer = RollingWindowBuffer(window_size=4, step_size=1, label_strategy="majority")
    for i, label in enumerate([0, 1, 1, 0]):
        buffer.add(np.array([i]), label=label)

    window = buffer.pop_window()
    assert window.y_window == 1
