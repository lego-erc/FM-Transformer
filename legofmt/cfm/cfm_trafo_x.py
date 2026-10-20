"""The flow's vector field: an x-transformers Encoder over the padded sequence.

Each token is embedded by a per-(mask, type, pdgid) conditional linear map: the
mean of three index-selected linear maps, computed as one matmul of the
one-hot-Kronecker-``x`` against the stacked weight banks (no ``(B, L, h, in)``
intermediate; its index backward was 60% of an eager step). The output
projection mirrors it. Time, and with ``step_cond`` the step size, enter as
sinusoidal embeddings passed to x-transformers as ``condition=`` for adaptive
RMSNorm. With one ``(t, d)`` per event the condition takes two values per event
(generated tokens, conditioning tokens), and the adaptive norms are handed those
two rows rather than ``L`` (``_per_event_gammas``).
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from x_transformers import ContinuousTransformerWrapper, Encoder
from x_transformers.x_transformers import AdaptiveLayerScale, AdaptiveRMSNorm
from legofmt.compat import fp32_attention, legacy_param_rename, needs_fp32_attention
from legofmt.cfm.geom_encoder import GeomEncoder


def _per_event_gammas(attn_layers: nn.Module) -> bool:
    """Let the adaptive norms / layer-scales take a packed ``(B, 3, D)`` condition: row 0
    the conditioning tokens' condition, row 1 the generated tokens', row 2's first ``L``
    entries the generated-token selector. Their ``to_gamma`` matmuls then run on 2 rows
    per event instead of ``L``: ~10% of a training step and of a solve (plate-Si config,
    H100). Any other condition goes through the original forward. Patches nothing and
    returns False when something else would consume the condition."""
    mods = [m for m in attn_layers.modules() if isinstance(m, (AdaptiveRMSNorm, AdaptiveLayerScale))]
    other = any(type(m).__name__.startswith("Adaptive") for m in attn_layers.modules() if m not in mods)
    if not mods or other or type(getattr(attn_layers, "adaptive_mlp", nn.Identity())).__name__ != "Identity":
        return False
    for m in mods:
        m.forward = partial(_packed_forward, m, m.forward)
    return True


def _packed_forward(mod, orig, x, *, condition, **kwargs):
    if condition.shape[1] != 3 or x.shape[1] == 3:
        return orig(x, condition=condition, **kwargs)
    sel = condition[:, 2, :x.shape[1], None] > 0.5
    g = mod.to_gamma(condition[:, :2])
    if isinstance(mod, AdaptiveRMSNorm):  # x_transformers.AdaptiveRMSNorm.forward
        return F.normalize(x, dim=-1) * mod.scale * (torch.where(sel, g[:, 1:2], g[:, :1]) + 1.)
    g = g.sigmoid()  # x_transformers.AdaptiveLayerScale.forward
    out = mod.fn(x, **kwargs)
    if isinstance(out, Tensor):
        return out * torch.where(sel, g[:, 1:2], g[:, :1])
    out, *rest = out
    return out * torch.where(sel, g[:, 1:2], g[:, :1]), *rest


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
        geom_encoder: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if step_cond and not time_cond:
            raise ValueError("step_cond=True requires time_cond=True")
        self.geom = None
        if geom_encoder is not None:
            g = dict(geom_encoder)
            n_prefix, self.geom_hide = g.pop("n_prefix"), g.pop("hide_scalars")
            # the composite cond-scalar rows (E_dep is row 1, the incoming particle row n_prefix)
            self.register_buffer("geom_scalar_rows", torch.tensor([0, *range(2, n_prefix)]), persistent=False)
            self.geom = GeomEncoder(g.pop("layers"), h_dim, **g)
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
        self._packed_cond = time_cond and _per_event_gammas(self.vf.attn_layers)

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

        self._register_load_state_dict_pre_hook(legacy_param_rename)

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
        else:
            self.global_cond = nn.Parameter(torch.zeros(1, h_dim))

    def _one_hot(self, mask: Tensor, types: Tensor, pdgids: Tensor, batch: int) -> Tensor:
        """``(B, L, nvtypes + ntypes + npdgids)`` one-hot over the three conditioning
        vocabularies, so the index-selected linear maps become one matmul.
        ``pdgids`` arrives as ``_F.pdgids``, i.e. ``(B, L, 1)``."""
        n = mask.shape[1]
        return torch.cat((
            nn.functional.one_hot(mask.reshape(batch, n).long(), self.nvtypes),
            nn.functional.one_hot(types.view(-1)[:n], self.ntypes).expand(batch, -1, -1),
            nn.functional.one_hot(pdgids.reshape(batch, n).long(), self.npdgids),
        ), dim=-1).to(self.cond_w_mask.dtype)

    def _embed(self, x: Tensor, oh: Tensor) -> Tensor:
        """Mean of the mask-, type- and pdgid-conditional up-projections of ``x``."""
        w  = torch.cat((self.cond_w_mask[:, 0], self.cond_w_types[:, 0], self.cond_w_pdgids[:, 0]))
        b  = torch.cat((self.cond_bi_mask, self.cond_bi_types, self.cond_bi_pdgids))
        xo = (oh.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)  # (B, L, K * in_dim)
        return (xo @ w.transpose(1, 2).reshape(-1, self.h_dim) + oh @ b) / 3

    def _project_out(self, h: Tensor, mask: Tensor, oh: Tensor) -> Tensor:
        """Mean of the three conditional down-projections; zero on conditioning slots."""
        w   = torch.cat((self.cond_w_mask[:, 1], self.cond_w_types[:, 1], self.cond_w_pdgids[:, 1]))
        b   = torch.cat((self.cond_bo_mask, self.cond_bo_types, self.cond_bo_pdgids))
        out = (h @ w.permute(1, 0, 2).reshape(self.h_dim, -1)).view(*h.shape[:-1], -1, self.in_dim)
        out = torch.einsum("blki,blk->bli", out, oh) + oh @ b
        return (mask == 1).unsqueeze(-1) * out / 3

    def _time_emb(self, t: Tensor, d: Tensor | None) -> Tensor:
        tf   = t.unsqueeze(-1) * self.freqs
        cond = torch.where(self.mask_freqs.bool(), tf.sin(), tf.cos())
        if self.step_cond and d is not None:
            cond = cond + self.step_gain * (d.unsqueeze(-1) * self.freqs_d).sin()
        return cond

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
        """``t`` / ``d`` per token ``(B, L)`` are used as given. Per event ``(B or 1, 1)``
        they condition the generated (``mask == 1``) tokens, and the conditioning tokens
        get ``t = 1, d = 0`` -- the values ``ProjectModel`` pins them to."""
        B, L = x.shape[:2]
        K = 0 if self.geom is None else self.geom.K
        cond_attn = c_con = None
        if not self.time_cond:
            cond = self.global_cond.expand(B, -1)
        elif t.shape[-1] == 1 and (d is None or d.shape[-1] == 1):
            gen   = (mask == 1).unsqueeze(-1)
            c_gen = self._time_emb(t, d).expand(B, 1, -1)
            c_con = self._time_emb(t.new_ones(1, 1), None if d is None else d.new_zeros(1, 1)).expand(B, 1, -1)
            cond  = torch.where(gen, c_gen, c_con)
            if self._packed_cond and 3 != L + K <= self.h_dim:
                # the selector spans the whole sequence; the K geometry tokens are conditioning tokens
                sel = F.pad(gen.squeeze(-1).to(cond.dtype), (K, self.h_dim - L - K)).unsqueeze(1)
                cond_attn = torch.cat((c_con, c_gen, sel), dim=1)
        else:
            cond = self._time_emb(t, d)

        oh   = self._one_hot(mask, types, pdgids, B)
        embd = self._embed(x, oh)
        if self.time_cond:
            embd = embd + cond
        if self.geom is not None:
            tok = self.geom().to(embd.dtype).expand(B, -1, -1)
            if self.time_cond:
                if c_con is None:
                    c_con = self._time_emb(t.new_ones(1, 1), None if d is None else d.new_zeros(1, 1)).expand(B, 1, -1)
                c_tok = c_con.expand(B, K, -1)
                tok = tok + c_tok
                cond = torch.cat((c_tok, cond), dim=1)
            embd = torch.cat((tok, embd), dim=1)
            if self.geom_hide:
                attn_mask = attn_mask.clone()
                attn_mask[:, self.geom_scalar_rows] = False
            attn_mask = torch.cat((attn_mask.new_ones(B, K), attn_mask), dim=1)

        # project_in / project_out are identities unless dim_in_out was set.
        embd = self.vf.emb_dropout(self.vf.project_in(embd))
        h = self.vf.project_out(self.vf.attn_layers(
            embd, mask=attn_mask, condition=cond if cond_attn is None else cond_attn))
        return self._project_out(h[:, K:], mask, oh)
