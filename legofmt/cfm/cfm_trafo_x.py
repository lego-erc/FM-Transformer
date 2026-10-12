"""The flow's vector field: an x-transformers Encoder over the padded sequence.

Each token is embedded by a per-(mask, type, pdgid) conditional linear map: the
mean of three index-selected linear maps, computed as one matmul of the
one-hot-Kronecker-``x`` against the stacked weight banks (no ``(B, L, h, in)``
intermediate; its index backward was 60% of an eager step). The output
projection mirrors it. Time, and with ``step_cond`` the step size, enter as
sinusoidal embeddings passed to x-transformers as ``condition=`` for adaptive
RMSNorm.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from x_transformers import ContinuousTransformerWrapper, Encoder
from legofmt.compat import fp32_attention, legacy_param_rename, needs_fp32_attention


SPECIES_R = 2.0   # species_flow: radius of the species vertices (base std 1)


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
        n_cond: int = 1,
        pt_dim: int = 64,
        sv_head: bool = False,
        sv_layers: int = 1,
        sv_dim: int | None = None,
        sv_chord: bool = False,
        ct_head: bool = False,
        ct_nmax: int = 0,
        sp_layers: int = 0,
        sp_dim: int | None = None,
        sp_block: int = 0,
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
        self.sp_block = sp_block
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
        if sp_layers > 0:   # MLP species head; the default linear one is a checkpoint-compatible special case
            w, layers = sp_dim or h_dim, []
            for i in range(sp_layers):
                layers += [nn.Linear(h_dim if i == 0 else w, w), nn.Mish()]
            self.species_head = nn.Sequential(*layers, nn.Linear(w, npdgids))
        else:
            self.species_head = nn.Linear(h_dim, npdgids)
        if sp_block > 0:   # species_flow: one vertex direction per real class (frozen at r I unless learned), flat so
            # muon_factory gives this embedding table AdamW; a checkpoint without it (count_s6) loads r I
            self.species_vertices = nn.Parameter(torch.eye(sp_block).flatten())
            self._register_load_state_dict_pre_hook(self._default_vertices)
        self.pt_head = nn.Sequential(
            nn.Linear(n_cond + in_dim + npdgids, pt_dim), nn.Mish(), nn.Linear(pt_dim, 1),
        )
        self.sv_chord = sv_chord
        if ct_head:  # per-event multiplicity over 0..ct_nmax: gate inputs + chord + linear energy + survival bit, deep
            n = n_cond + in_dim + npdgids + 3
            self.ct_head = nn.Sequential(nn.Linear(n, 256), nn.Mish(), nn.Linear(256, 256), nn.Mish(),
                                         nn.Linear(256, 256), nn.Mish(), nn.Linear(256, 256), nn.Mish(),
                                         nn.Linear(256, ct_nmax + 1))
        if sv_head:  # "the leading particle is the primary's species", the gate's inputs (+ the chord)
            w, n = sv_dim or pt_dim, n_cond + in_dim + npdgids + int(sv_chord)
            layers = []
            for i in range(sv_layers):
                layers += [nn.Linear(n if i == 0 else w, w), nn.Mish()]
            self.sv_head = nn.Sequential(*layers, nn.Linear(w, 1))

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
            self._register_load_state_dict_pre_hook(legacy_param_rename)
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

    def _default_vertices(self, state_dict, prefix, *_) -> None:
        state_dict.setdefault(prefix + "species_vertices", torch.eye(self.sp_block).flatten())

    def vertices(self) -> Tensor:
        """``(d, d)``, row k = class k + 1's vertex: the learned directions on the sphere of radius r (fixed, so
        the margin against the unit base is fixed and the vertices cannot shrink toward the base mean)."""
        return SPECIES_R * nn.functional.normalize(self.species_vertices.float().view(self.sp_block, -1), dim=-1)

    def _soft_label(self, z: Tensor, t: Tensor, C: Tensor) -> Tensor:
        """``(1 - t) e_0 + t softmax(beta z . C)``: the block-only likelihood ratio over the real classes on the
        path ``z_t = (1 - t) z_0 + t c_s`` (base std 1, equal vertex norms, ``beta = t / (1 - t)^2``), mixed with
        today's "unknown" label so that t = 0 is exactly class 0 and t -> 1 the hard one-hot of the nearest
        vertex. fp32: beta reaches 1e6."""
        with torch.autocast(z.device.type, enabled=False):
            tt  = t.float().unsqueeze(-1)
            omt = 1.0 - tt
            p   = torch.softmax((z.float() @ C.t()) * (tt / omt.clamp(min=1e-3) ** 2), dim=-1)
            return torch.cat((omt, tt * p), dim=-1)

    def _block_velocity(self, logits: Tensor, z: Tensor, t: Tensor, C: Tensor) -> Tensor:
        """``(p C - z_t) / (1 - t)`` with ``p`` the posterior over the real classes: on the linear path the
        conditional velocity is ``(c_s - z_t) / (1 - t)``, so this is the marginal flow-matching velocity at the
        true posterior, and the species cross-entropy is the block's transport loss."""
        with torch.autocast(z.device.type, enabled=False):
            p = torch.softmax(logits[..., 1:].float(), dim=-1)
            return (p @ C - z.float()) / (1.0 - t.float()).clamp(min=1e-3).unsqueeze(-1)

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

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
        t: Tensor | None = None,
        d: Tensor | None = None,
        return_species: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.time_cond:
            tf = t.unsqueeze(-1) * self.freqs
            cond = torch.where(self.mask_freqs.bool(), tf.sin(), tf.cos())
            if self.step_cond and d is not None:
                cond = cond + self.step_gain * (d.unsqueeze(-1) * self.freqs_d).sin()
        else:
            cond = self.global_cond.expand(x.shape[0], -1)

        if self.sp_block > 0:   # species_flow: a generated row carries -1 and reads a soft label off its block
            x, z = x[..., :self.in_dim], x[..., self.in_dim:]
            C    = self.vertices().detach()   # through the network the vertices collapse; _step's own loss moves them
            gen  = (pdgids.reshape(mask.shape) < 0).unsqueeze(-1)
            oh   = self._one_hot(mask, types, pdgids.clamp(min=0), x.shape[0])
            oh[..., -self.npdgids:] = torch.where(gen, self._soft_label(z, t, C), oh[..., -self.npdgids:])
        else:
            oh   = self._one_hot(mask, types, pdgids, x.shape[0])
        embd = self._embed(x, oh)
        if self.time_cond:
            embd = embd + cond

        # project_in / project_out are identities unless dim_in_out was set.
        embd = self.vf.project_in(embd)
        if self.training:
            embd = self.vf.emb_dropout(embd)
        h = self.vf.project_out(self.vf.attn_layers(embd, mask=attn_mask, condition=cond))
        v = self._project_out(h, mask, oh)
        if self.sp_block > 0:   # the block's velocity rides the species head's posterior, on generated rows only
            sp = self.species_head(h)
            v  = torch.cat((v, gen * self._block_velocity(sp, z, t, C)), dim=-1)
            return (v, sp) if return_species else v
        return (v, self.species_head(h)) if return_species else v
