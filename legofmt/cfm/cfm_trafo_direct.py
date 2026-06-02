"""Direct (no-time) mirror of :class:`legofmt.cfm.cfm_trafo_x.CFMTrafo_x`.

Same factorized projections, same x-transformer Encoder, but the model
predicts the residual ``target - base`` at the generated slots instead
of a velocity field. The flow-time argument and its sinusoidal embedding
are removed; a single learnable ``global_cond`` vector replaces the
time-derived condition fed to adaptive-RMSNorm so the Encoder
architecture stays identical to the velocity variant. The Euler step
``final = base + residual`` and the manifold snap are applied in the
:class:`legofmt.main.modules_direct.ProjectModel` wrapper.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from x_transformers import ContinuousTransformerWrapper, Encoder


class CFMTrafo_x(nn.Module):
    """Endpoint-prediction transformer with factorized projections.

    Same class name as :class:`legofmt.cfm.cfm_trafo_x.CFMTrafo_x` (the
    velocity variant); the module path is the disambiguator.
    """

    def __init__(
        self,
        h_dim: int = 256,
        *,
        nhead: int = 8,
        max_seq_l: int = 9,
        nvtypes: int = 3,
        ntypes: int | None = None,
        in_dim: int = 6,
        ff_mult: int = 1,
        dropout: float = 0.1,
        nlayers: int = 4,
        xavier_gain: float = 1.0,
        npdgids: int = 1,
        dim_in_out: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        ntypes = ntypes if ntypes is not None else max_seq_l
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.max_seq_l = max_seq_l
        self.nvtypes = nvtypes
        self.ntypes = ntypes
        self.npdgids = npdgids

        self.vf = ContinuousTransformerWrapper(
            dim_in=dim_in_out,
            dim_out=dim_in_out,
            max_seq_len=max_seq_l,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Encoder(
                dim=h_dim,
                depth=nlayers,
                heads=nhead,
                attn_dropout=dropout,
                ff_dropout=dropout,
                ff_mult=ff_mult,
                dim_condition=h_dim,
                **kwargs,
            ),
        )

        self.cond_w_mask    = nn.Parameter(torch.empty(nvtypes, 2, h_dim, in_dim))
        self.cond_bi_mask   = nn.Parameter(torch.empty(nvtypes, h_dim))
        self.cond_bo_mask   = nn.Parameter(torch.empty(nvtypes, in_dim))
        self.cond_w_types   = nn.Parameter(torch.empty(ntypes,  2, h_dim, in_dim))
        self.cond_bi_types  = nn.Parameter(torch.empty(ntypes,  h_dim))
        self.cond_bo_types  = nn.Parameter(torch.empty(ntypes,  in_dim))
        self.cond_w_pdgids  = nn.Parameter(torch.empty(npdgids, 2, h_dim, in_dim))
        self.cond_bi_pdgids = nn.Parameter(torch.empty(npdgids, h_dim))
        self.cond_bo_pdgids = nn.Parameter(torch.empty(npdgids, in_dim))

        for p in (
            self.cond_w_mask, self.cond_bi_mask, self.cond_bo_mask,
            self.cond_w_types, self.cond_bi_types, self.cond_bo_types,
            self.cond_w_pdgids, self.cond_bi_pdgids, self.cond_bo_pdgids,
        ):
            nn.init.xavier_normal_(p, gain=xavier_gain)

        # Learnable replacement for the sinusoidal time embedding so the
        # Encoder's adaptive-RMSNorm / layerscale paths still receive a
        # condition signal.
        self.global_cond = nn.Parameter(torch.zeros(1, h_dim))

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
    ) -> Tensor:
        n = x.shape[1]
        b = x.shape[0]
        mi, ti, pi = mask.view(-1), types.view(-1)[:n], pdgids.view(-1)
        s3 = (-1, n, self.h_dim)
        so = (-1, n, self.in_dim)
        s4 = (-1, n, self.h_dim, self.in_dim)

        embd = (
            torch.einsum(
                "ijl,ijkl->ijk", x,
                self.cond_w_mask  [mi, 0].view(s4)
              + self.cond_w_types [ti, 0]
              + self.cond_w_pdgids[pi, 0].view(s4),
            )
          + self.cond_bi_mask  [mi].view(s3)
          + self.cond_bi_types [ti].view(s3)
          + self.cond_bi_pdgids[pi].view(s3)
        ) / 3

        cond = self.global_cond.expand(b, -1)

        embd = self.vf.project_in(embd)
        if self.training:
            embd = self.vf.emb_dropout(embd)
        h = self.vf.project_out(self.vf.attn_layers(embd, mask=attn_mask, condition=cond))

        return (mask == 1).unsqueeze(-1) * (
            torch.einsum(
                "ijk,ijkl->ijl", h,
                self.cond_w_mask  [mi, 1].view(s4)
              + self.cond_w_types [ti, 1]
              + self.cond_w_pdgids[pi, 1].view(s4),
            )
          + self.cond_bo_mask  [mi].view(so)
          + self.cond_bo_types [ti].view(so)
          + self.cond_bo_pdgids[pi].view(so)
        ) / 3
