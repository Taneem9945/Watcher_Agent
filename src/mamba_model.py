from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
VENDOR_MAMBA = ROOT / "vendor" / "mamba"
if VENDOR_MAMBA.exists() and str(VENDOR_MAMBA) not in sys.path:
    sys.path.insert(0, str(VENDOR_MAMBA))

from mamba_ssm import Mamba  # type: ignore

MAMBA_AVAILABLE = True
MAMBA_SOURCE = "official-vendored"


class MambaWatcher(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__()

        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def _pool_hidden(self, h, mask=None):
        if mask is None:
            if self.pooling == "last":
                return h[:, -1, :]
            if self.pooling == "mean":
                return h.mean(dim=1)
        else:
            mask = mask.to(device=h.device, dtype=h.dtype)
            if mask.ndim != 2:
                raise ValueError("mask must have shape [batch_size, sequence_length]")
            if mask.shape != h.shape[:2]:
                raise ValueError("mask shape must match hidden sequence shape")
            lengths = mask.sum(dim=1).clamp(min=1)
            if self.pooling == "last":
                last_indices = (lengths.long() - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])
                return h.gather(dim=1, index=last_indices).squeeze(1)
            if self.pooling == "mean":
                return (h * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def encode(self, x, mask=None):
        """
        Return pooled sequence representations for downstream signal building.

        x shape: [batch_size, sequence_length, input_dim]
        mask shape, optional: [batch_size, sequence_length]
        """
        x = self.input_proj(x)
        x = self.dropout(x)
        h = self.mamba(x)
        return self._pool_hidden(h, mask=mask)

    def forward(self, x, mask=None):
        """
        x shape: [batch_size, sequence_length, input_dim]
        """
        h = self.encode(x, mask=mask)
        logits = self.classifier(h)
        return logits
