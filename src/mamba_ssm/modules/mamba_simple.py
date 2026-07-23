from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class Mamba(nn.Module):
    """
    Local Mamba-1 style block with a pure PyTorch selective scan fallback.

    This keeps the public interface small and editable inside the workspace.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        **_: object,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        nn.init.uniform_(self.dt_proj.bias, -0.1, 0.1)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = torch.log(A).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(A.to(**factory_kwargs))
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError(
                "The local vendored Mamba block implements full-sequence forward only."
            )

        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :seqlen]
        x = self.act(x)
        x = x.transpose(1, 2)

        params = self.x_proj(x)
        dt, B, C = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(dt).transpose(1, 2)
        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            u=x.transpose(1, 2),
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D.float(),
            z=z.transpose(1, 2),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = y.transpose(1, 2)
        return self.out_proj(y)
