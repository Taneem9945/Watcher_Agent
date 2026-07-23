from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    if B.dim() == 2:
        is_variable_B = False
        B = B.float()
    elif B.dim() == 3:
        is_variable_B = True
        B = B.float()
    else:
        raise ValueError("B must be rank-2 or rank-3")

    if C.dim() == 2:
        is_variable_C = False
        C = C.float()
    elif C.dim() == 3:
        is_variable_C = True
        C = C.float()
    else:
        raise ValueError("C must be rank-2 or rank-3")

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    return selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


def mamba_inner_fn(*args, **kwargs):
    raise NotImplementedError(
        "The vendored prototype uses the plain selective_scan_fn path."
    )
