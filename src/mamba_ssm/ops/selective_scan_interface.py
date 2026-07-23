from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
):
    """
    Pure PyTorch selective scan used by the local Mamba-1 style block.

    Shapes:
    - u: [batch, d_inner, length]
    - delta: [batch, d_inner, length]
    - A: [d_inner, d_state]
    - B: [batch, length, d_state]
    - C: [batch, length, d_state]
    """
    if u.ndim != 3 or delta.ndim != 3:
        raise ValueError("u and delta must have shape [batch, d_inner, length]")

    batch, d_inner, length = u.shape
    d_state = A.shape[-1]

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    outputs = []
    B = B.to(dtype=u.dtype)
    C = C.to(dtype=u.dtype)
    A = A.to(dtype=u.dtype)

    for t in range(length):
        dt = delta[:, :, t].unsqueeze(-1)
        decay = torch.exp(dt * A.unsqueeze(0))
        drive = B[:, t, :].unsqueeze(1) * u[:, :, t].unsqueeze(-1)
        state = decay * state + drive
        y = torch.sum(state * C[:, t, :].unsqueeze(1), dim=-1)
        if D is not None:
            y = y + D.view(1, -1) * u[:, :, t]
        if z is not None:
            y = y * F.silu(z[:, :, t])
        outputs.append(y)

    return torch.stack(outputs, dim=-1)


def mamba_inner_fn(*args, **kwargs):
    return selective_scan_fn(*args, **kwargs)
