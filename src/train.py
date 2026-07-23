from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .metrics import accuracy, false_positive_rate, f1_score, precision, recall
from .utils import ensure_dir


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    val_precision: list[float]
    val_recall: list[float]
    val_f1: list[float]
    val_fpr: list[float]


def _make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(np.asarray(X), dtype=torch.float32),
        torch.tensor(np.asarray(y), dtype=torch.long),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def _run_eval(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    all_true = []
    all_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_true.extend(yb.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy(all_true, all_pred),
        "precision": precision(all_true, all_pred),
        "recall": recall(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "fpr": false_positive_rate(all_true, all_pred),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device,
):
    """
    Train the MambaWatcher model.

    Requirements:
    - Use CrossEntropyLoss.
    - Use AdamW.
    - Track train loss and validation loss.
    - Track validation accuracy, precision, recall, F1, and false positive rate.
    - Save best model based on validation F1.
    - Support early stopping.
    """
    training_cfg = config.get("training", {})
    lr = float(training_cfg.get("learning_rate", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    epochs = int(training_cfg.get("epochs", 20))
    patience = int(training_cfg.get("early_stopping_patience", 5))
    ckpt_path = Path("checkpoints/best_mamba_watcher.pt")
    ensure_dir(ckpt_path.parent)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = TrainingHistory([], [], [], [], [], [], [])
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    model.to(device)

    for _epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in tqdm(train_loader, leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_metrics = _run_eval(model, val_loader, device)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_metrics["loss"])
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_precision.append(val_metrics["precision"])
        history.val_recall.append(val_metrics["recall"])
        history.val_f1.append(val_metrics["f1"])
        history.val_fpr.append(val_metrics["fpr"])

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_f1": best_f1,
            }
            torch.save(best_state, ckpt_path)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    return {
        "history": history.__dict__,
        "best_val_f1": best_f1,
        "checkpoint_path": str(ckpt_path),
    }


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):
    return (
        _make_loader(X_train, y_train, batch_size=batch_size, shuffle=True),
        _make_loader(X_val, y_val, batch_size=batch_size, shuffle=False),
        _make_loader(X_test, y_test, batch_size=batch_size, shuffle=False),
    )
