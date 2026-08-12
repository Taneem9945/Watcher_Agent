from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .context_builder import _risk_level


def _prediction_label(prediction: int) -> str:
    return "attack" if int(prediction) == 1 else "benign"


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2_norm": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
    }


def _top_activations(values: torch.Tensor, top_k: int = 8) -> list[dict[str, float | int]]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return []
    top_k = min(int(top_k), int(flat.numel()))
    magnitudes = torch.abs(flat)
    indices = torch.topk(magnitudes, k=top_k).indices.tolist()
    return [
        {
            "index": int(idx),
            "value": float(flat[idx].item()),
            "magnitude": float(magnitudes[idx].item()),
        }
        for idx in indices
    ]


def build_mamba_signal_packet(
    *,
    window_id: int,
    logits: torch.Tensor,
    pooled_embedding: torch.Tensor,
    true_label: int | None = None,
    previous_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert raw Mamba inference outputs into a structured signal packet.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")
    if pooled_embedding.ndim != 2:
        raise ValueError("pooled_embedding must have shape [batch_size, hidden_dim]")
    if logits.shape[0] != pooled_embedding.shape[0]:
        raise ValueError("logits and pooled_embedding batch sizes must match")
    if logits.shape[1] != 2:
        raise ValueError("This prototype expects binary classification logits with 2 classes")

    probs = torch.softmax(logits, dim=1)
    attack_probs = probs[:, 1]
    benign_probs = probs[:, 0]
    predictions = torch.argmax(logits, dim=1)
    margins = logits[:, 1] - logits[:, 0]

    signal_rows: list[dict[str, Any]] = []
    for i in range(logits.shape[0]):
        embedding_row = pooled_embedding[i]
        attack_probability = float(attack_probs[i].item())
        benign_probability = float(benign_probs[i].item())
        prediction = int(predictions[i].item())
        signal_rows.append(
            {
                "window_id": int(window_id),
                "row_index": int(i),
                "prediction": prediction,
                "prediction_label": _prediction_label(prediction),
                "attack_probability": attack_probability,
                "benign_probability": benign_probability,
                "confidence": float(max(attack_probability, benign_probability)),
                "logit_margin": float(margins[i].item()),
                "risk_level": _risk_level(attack_probability),
                "embedding_summary": _tensor_stats(embedding_row),
                "top_embedding_activations": _top_activations(embedding_row, top_k=8),
                "true_label": None if true_label is None else int(true_label),
            }
        )

    packet: dict[str, Any] = {
        "window_id": int(window_id),
        "batch_size": int(logits.shape[0]),
        "num_classes": int(logits.shape[1]),
        "signal_rows": signal_rows,
    }
    if previous_signal is not None:
        packet["previous_signal"] = previous_signal
    return packet


def build_mamba_encoder_signal_packet(
    *,
    window_ids: list[int],
    pooled_embedding: torch.Tensor,
    metadata: list[dict[str, Any]] | None = None,
    previous_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert Mamba encoder outputs into representation-only signal rows.

    This packet intentionally excludes classifier verdicts such as prediction,
    attack_probability, and risk_level.
    """
    if pooled_embedding.ndim != 2:
        raise ValueError("pooled_embedding must have shape [batch_size, hidden_dim]")
    if len(window_ids) != pooled_embedding.shape[0]:
        raise ValueError("window_ids length must match pooled_embedding batch size")
    if metadata is not None and len(metadata) != pooled_embedding.shape[0]:
        raise ValueError("metadata length must match pooled_embedding batch size")

    signal_rows: list[dict[str, Any]] = []
    for i in range(pooled_embedding.shape[0]):
        row: dict[str, Any] = {
            "window_id": int(window_ids[i]),
            "row_index": int(i),
            "signal_type": "mamba_encoder_representation",
            "embedding_summary": _tensor_stats(pooled_embedding[i]),
            "top_embedding_activations": _top_activations(pooled_embedding[i], top_k=8),
        }
        if metadata is not None:
            row["window_metadata"] = dict(metadata[i])
        signal_rows.append(row)

    packet: dict[str, Any] = {
        "batch_size": int(pooled_embedding.shape[0]),
        "signal_type": "mamba_encoder_representation",
        "signal_rows": signal_rows,
    }
    if previous_signal is not None:
        packet["previous_signal"] = previous_signal
    return packet


@torch.no_grad()
def infer_window_signal(
    model,
    X_window,
    window_id: int,
    true_label: int | None = None,
    device: torch.device | str = "cpu",
    previous_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single window through the model and return a richer signal packet.
    """
    model.eval()
    x = torch.tensor(np.asarray(X_window), dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError("X_window must be 2D or 3D")

    pooled_embedding = model.encode(x)
    logits = model.classifier(pooled_embedding)
    return build_mamba_signal_packet(
        window_id=window_id,
        logits=logits,
        pooled_embedding=pooled_embedding,
        true_label=true_label,
        previous_signal=previous_signal,
    )


@torch.no_grad()
def infer_window_encoder_signal(
    model,
    X_window,
    window_id: int,
    mask=None,
    device: torch.device | str = "cpu",
    metadata: dict[str, Any] | None = None,
    previous_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single window through the Mamba encoder and return representation-only signals.
    """
    model.eval()
    x = torch.tensor(np.asarray(X_window), dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError("X_window must be 2D or 3D")

    mask_tensor = None
    if mask is not None:
        mask_tensor = torch.tensor(np.asarray(mask), dtype=torch.float32, device=device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim != 2:
            raise ValueError("mask must be 1D or 2D")

    pooled_embedding = model.encode(x, mask=mask_tensor)
    return build_mamba_encoder_signal_packet(
        window_ids=[int(window_id)],
        pooled_embedding=pooled_embedding,
        metadata=[metadata or {}],
        previous_signal=previous_signal,
    )
