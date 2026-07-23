from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .metrics import accuracy, confusion_matrix, false_positive_rate, f1_score, precision, recall


def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model.

    Required metrics:
    - accuracy
    - precision
    - recall
    - F1
    - false positive rate
    - confusion matrix
    - average inference latency per window
    - throughput in windows per second
    """
    model.eval()
    all_true = []
    all_pred = []
    sample_latencies = []
    total_windows = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            t0 = time.perf_counter()
            logits = model(xb)
            batch_elapsed = time.perf_counter() - t0
            sample_latencies.append(batch_elapsed / max(xb.shape[0], 1))
            preds = torch.argmax(logits, dim=1)
            all_true.extend(yb.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())
            total_windows += xb.shape[0]

    elapsed = max(time.perf_counter() - start_time, 1e-12)
    avg_latency = float(np.mean(sample_latencies)) if sample_latencies else 0.0
    throughput = float(total_windows / elapsed)

    return {
        "accuracy": accuracy(all_true, all_pred),
        "precision": precision(all_true, all_pred),
        "recall": recall(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "false_positive_rate": false_positive_rate(all_true, all_pred),
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        "average_inference_latency_per_window": avg_latency,
        "throughput_windows_per_second": throughput,
    }
