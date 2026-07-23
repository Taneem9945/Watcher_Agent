from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    denom = tp + fp
    return float(tp / denom) if denom else 0.0


def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    denom = tp + fn
    return float(tp / denom) if denom else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return float((2 * p * r) / denom) if denom else 0.0


def false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    denom = fp + tn
    return float(fp / denom) if denom else 0.0
