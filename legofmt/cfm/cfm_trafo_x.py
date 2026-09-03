from __future__ import annotations

import warnings
from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from x_transformers import ContinuousTransformerWrapper, Encoder
from x_transformers.attend import Attend


# bf16 rounding of unit q, k perturbs cos(theta) by ~2**-8, i.e. the logits by
# ~qk_norm_scale/256: negligible at the library default 10, but O(1) for the
# pre-2.25.5 checkpoints whose effective scale is 1000 (0.27 -> 0.03 rel. error
# on kin_020926 with fp32 attention). Only those get the fp32 kernel.
FP32_ATTN_QK_SCALE = 100


def needs_fp32_attention(model_args: dict) -> bool:
    return bool(model_args.get("attn_qk_norm")) and model_args.get("attn_qk_norm_scale", 10) > FP32_ATTN_QK_SCALE


def fp32_attention(module: nn.Module) -> None:
    """Run every ``Attend`` in fp32 even under autocast (see ``needs_fp32_attention``)."""
    def _wrap(fwd):
        def forward(q, k, v, *args, **kwargs):
            with torch.autocast(q.device.type, enabled=False):
                return fwd(q.float(), k.float(), v.float(), *args, **kwargs)
        return forward

    for m in module.modules():
        if isinstance(m, Attend):
            m.forward = _wrap(m.forward)


# Pre-refactor parameter names -> current names.
_LEGACY_RENAME: dict[str, str] = {
    "l_mask_":    "cond_w_mask",
    "b_mask_":    "cond_bi_mask",
    "bo_mask_":   "cond_bo_mask",
    "l_types_":   "cond_w_types",
    "b_types_":   "cond_bi_types",
    "bo_types_":  "cond_bo_types",
    "l_pdgids_":  "cond_w_pdgids",
    "b_pdgids_":  "cond_bi_pdgids",
    "bo_pdgids_": "cond_bo_pdgids",
}


class CFMTrafo_x(nn.Module):

    def __init__(
        self,
        h_dim: int = 256,
        *,
        nhead: int = 8,
        max_seq_l: int = 9,
        nvtypes: int = 2,
        ntypes: int | None = 4,
        in_dim: int = 6,
        ff_mult: int = 1,
        dropout: float = 0.1,
        nlayers: int = 4,
        xavier_gain: float = 1.0,
        npdgids: int = 1,
        dim_in_out: int | None = None,
        time_cond: bool = True,
        step_cond: bool = False,
        grad_ckpt: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if step_cond and not time_cond:
            raise ValueError("step_cond=True requires time_cond=True")
        ntypes = ntypes if ntypes is not None else max_seq_l
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.max_seq_l = max_seq_l
        self.nvtypes = nvtypes
        self.ntypes = ntypes
        self.npdgids = npdgids
        self.time_cond = time_cond
        self.step_cond = step_cond

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
        if needs_fp32_attention(kwargs):
            fp32_attention(self.vf)

        if grad_ckpt:
            for _norms, _block, _residual in self.vf.attn_layers.layers:
                _block.forward = partial(checkpoint, _block.forward, use_reentrant=False)

        # Per conditioning source: [:, 0] up-projection, [:, 1] down-projection.
        self.cond_w_mask    = nn.Parameter(torch.empty(nvtypes, 2, h_dim, in_dim))
        self.cond_bi_mask   = nn.Parameter(torch.empty(nvtypes, h_dim))
        self.cond_bo_mask   = nn.Parameter(torch.empty(nvtypes, in_dim))
        self.cond_w_types   = nn.Parameter(torch.empty(ntypes,  2, h_dim, in_dim))
        self.cond_bi_types  = nn.Parameter(torch.empty(ntypes,  h_dim))
        self.cond_bo_types  = nn.Parameter(torch.empty(ntypes,  in_dim))
        self.cond_w_pdgids  = nn.Parameter(torch.empty(npdgids, 2, h_dim, in_dim))
        self.cond_bi_pdgids = nn.Parameter(torch.empty(npdgids, h_dim))
        self.cond_bo_pdgids = nn.Parameter(torch.empty(npdgids, in_dim))

        w_std = xavier_gain * 3 ** 0.5 * (2.0 / (in_dim + h_dim)) ** 0.5
        for p in (self.cond_w_mask, self.cond_w_types, self.cond_w_pdgids):
            nn.init.normal_(p, std=w_std)
        for p in (
            self.cond_bi_mask, self.cond_bi_types, self.cond_bi_pdgids,
            self.cond_bo_mask, self.cond_bo_types, self.cond_bo_pdgids,
        ):
            nn.init.zeros_(p)

        if time_cond:
            # Sinusoidal time embedding, freqs scaled by h_dim.
            self.register_buffer("freqs", h_dim * 1e-4 ** (torch.arange(h_dim) / h_dim))
            self.register_buffer("mask_freqs", torch.arange(h_dim) % 2)
            if step_cond:
                self.register_buffer(
                    "freqs_d", 2 * torch.pi * 2 ** (torch.arange(h_dim) * 3.0 / h_dim),
                    persistent=False,
                )
                self.step_gain = nn.Parameter(torch.zeros(1))
            self._register_load_state_dict_pre_hook(self._legacy_param_rename)
        else:
            self.global_cond = nn.Parameter(torch.zeros(1, h_dim))

    @staticmethod
    def _legacy_param_rename(state_dict, prefix, *_):
        renames = {
            k: prefix + _LEGACY_RENAME[suf]
            for k in list(state_dict)
            if k.startswith(prefix) and (suf := k[len(prefix):]) in _LEGACY_RENAME
        }
        if not renames:
            return
        warnings.warn(
            "Remapping legacy CFMTrafo_x parameter keys "
            "(e.g. 'l_mask_' -> 'cond_w_mask'); re-save to silence.",
            DeprecationWarning, stacklevel=4,
        )
        for old, new in renames.items():
            state_dict[new] = state_dict.pop(old)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
        t: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        n = x.shape[1]
        mi, ti, pi = mask.view(-1), types.view(-1)[:n], pdgids.view(-1)
        s3 = (-1, n, self.h_dim)
        so = (-1, n, self.in_dim)
        s4 = (-1, n, self.h_dim, self.in_dim)

        if self.time_cond:
            tf = t.unsqueeze(-1) * self.freqs
            cond = torch.where(self.mask_freqs.bool(), tf.sin(), tf.cos())
            if self.step_cond and d is not None:
                cond = cond + self.step_gain * (d.unsqueeze(-1) * self.freqs_d).sin()
        else:
            cond = self.global_cond.expand(x.shape[0], -1)

        # Up-projection: the three source-indexed weights are summed
        # before the einsum (single fused contraction); biases summed
        # likewise; divide by 3 to average. Inlined to let intermediates
        # be freed before the next op.
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
        if self.time_cond:
            embd = embd + cond

        # project_in / project_out are identities unless dim_in_out was set.
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
