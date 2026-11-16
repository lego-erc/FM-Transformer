"""The flow's vector field: a transformer over the event's padded token sequence.

```
tokens     [z | g_1 .. g_K | x_1 .. x_L]               the event's latent and the geometry tokens are optional

embed      h_i = mean_k(W_k[c_ik] x_i + b_k[c_ik])     k: the token's mask, type and pdgid; one linear map per value
           h_z = lat_in z,   h_g = GeomEncoder()

time       h_i = h_i + emb_i,   emb_i = emb(t, d) on a generated token, emb(1, 0) on a conditioning one
           emb(t, d) = sinusoid(t) + step_gain sin(d freqs_d)

encoder    h = Encoder(h | emb)                         every block's adaptive RMSNorm / layer scale reads emb_i

read-out   v_i = mean_k(V_k[c_ik] h_i + a_k[c_ik])      on a generated token, 0 on a conditioning one
           v_z = lat_out h_z
```

A generated token (``mask == 1``) is one the flow moves: the outgoing particles, the E_dep token, the latent. A
conditioning token sits at the data and is not transported, which is ``t = 1, d = 0``: the scalars, the incoming
particle, the geometry tokens. ``(t, d)`` is one time and one step size per event.

Three parts are optional:

    step_cond      the step size ``d`` of a flow-map step enters the time embedding
    geom_encoder   K summary tokens of a layered stack (``geom_encoder.py``), prepended as conditioning tokens
    lat_dim        the event's latent ``z`` (``model_conf.edep_ae``), prepended as one generated token

The encoder is an x-transformers ``Encoder``. Everything else (``time_cond: false``, per-token times,
``dim_in_out``, ``grad_ckpt``) is :mod:`legofmt.legacy.cfm_trafo_x`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from x_transformers import ContinuousTransformerWrapper, Encoder

from legofmt.cfm.geom_encoder import GeomEncoder
from legofmt.cfm.event_cond import per_event_condition, per_event_gammas
from legofmt.compat import fp32_attention, legacy_param_rename, needs_fp32_attention
from legofmt.legacy.cfm_trafo_x import LegacyField


def _prepend(tokens: Tensor, c: Tensor, generated: bool, h: Tensor, emb: Tensor, gen: Tensor, attn_mask: Tensor):
    """Prepends ``tokens`` (B, n, D) to the sequence ``(h, emb, gen, attn_mask)``.

    They get the time embedding ``c`` (B, 1, D), are attended, and are generated or conditioning tokens.
    """
    n = tokens.shape[1]
    c = c.expand(-1, n, -1)
    return (torch.cat((tokens + c, h), dim=1), torch.cat((c, emb), dim=1),
            F.pad(gen, (n, 0), value=generated), F.pad(attn_mask, (n, 0), value=True))


class CFMTrafo_x(LegacyField):

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
        lat_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        ntypes = ntypes if ntypes is not None else max_seq_l
        self.h_dim     = h_dim
        self.in_dim    = in_dim
        self.max_seq_l = max_seq_l
        self.nvtypes   = nvtypes
        self.ntypes    = ntypes
        self.npdgids   = npdgids
        self.step_cond = step_cond
        self.lat_dim   = lat_dim

        # geometry tokens (model_args.geom_encoder): they stand in for the cond-scalar tokens, which hide_scalars
        # masks out of the attention (E_dep is token 1, the incoming particle token n_prefix)
        self.geom = None
        if geom_encoder is not None:
            g = dict(geom_encoder)
            n_prefix, self.geom_hide = g.pop("n_prefix"), g.pop("hide_scalars")
            self.register_buffer("geom_scalar_tokens", torch.tensor([0, *range(2, n_prefix)]), persistent=False)
            self.geom = GeomEncoder(g.pop("layers"), h_dim, **g)

        # the encoder
        self.vf = ContinuousTransformerWrapper(
            dim_in=dim_in_out,       # legacy; None: no map
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
        self._per_event_cond = time_cond and per_event_gammas(self.vf.attn_layers)

        # token embedding and read-out: per value of the mask, the type and the pdgid a linear map in ([:, 0]: W, bi: b)
        # and one out ([:, 1]: V, bo: a)
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

        # the latent token (model_conf.edep_ae); a zero read-out: an untrained token moves nothing
        if lat_dim:
            self.lat_in  = nn.Linear(lat_dim, h_dim)
            self.lat_out = nn.Linear(h_dim, lat_dim)
            nn.init.zeros_(self.lat_out.weight)
            nn.init.zeros_(self.lat_out.bias)

        w_std = xavier_gain * 3 ** 0.5 * (2.0 / (in_dim + h_dim)) ** 0.5
        for p in (self.cond_w_mask, self.cond_w_types, self.cond_w_pdgids):
            nn.init.normal_(p, std=w_std)
        for p in (
            self.cond_bi_mask, self.cond_bi_types, self.cond_bi_pdgids,
            self.cond_bo_mask, self.cond_bo_types, self.cond_bo_pdgids,
        ):
            nn.init.zeros_(p)

        # time embedding: sinusoids of t, their frequencies scaled by h_dim; with step_cond a second set for d,
        # behind a gain that starts at zero
        self.register_buffer("freqs", h_dim * 1e-4 ** (torch.arange(h_dim) / h_dim))
        self.register_buffer("mask_freqs", torch.arange(h_dim) % 2)
        if step_cond:
            self.register_buffer("freqs_d", 2 * torch.pi * 2 ** (torch.arange(h_dim) * 3.0 / h_dim), persistent=False)
            self.step_gain = nn.Parameter(torch.zeros(1))

        self._init_legacy(time_cond, dim_in_out, grad_ckpt)

    def _one_hot(self, mask: Tensor, types: Tensor, pdgids: Tensor) -> Tensor:
        """Builds each token's one-hot over the three vocabularies: its mask value, its type, its pdgid.

        The index-selected linear maps become one matmul on it.

        Args:
            mask:   (B, L), 1 on generated tokens.
            types:  (1, L), each position's type.
            pdgids: (B, L, 1), each token's species index (``_F.pdgids``).

        Returns:
            (B, L, nvtypes + ntypes + npdgids).
        """
        B, L = mask.shape[:2]
        return torch.cat((
            F.one_hot(mask.reshape(B, L).long(), self.nvtypes),
            F.one_hot(types.view(-1)[:L], self.ntypes).expand(B, -1, -1),
            F.one_hot(pdgids.reshape(B, L).long(), self.npdgids),
        ), dim=-1).to(self.cond_w_mask.dtype)

    def _embed(self, x: Tensor, oh: Tensor) -> Tensor:
        """Embeds the tokens ``x``: ``h_i = mean_k(W_k[c_ik] x_i + b_k[c_ik])``.

        One matmul of the Kronecker product ``oh (x) x`` against the stacked maps: no ``(B, L, h, in)``
        intermediate, whose index backward was 60% of an eager step.
        """
        w  = torch.cat((self.cond_w_mask[:, 0], self.cond_w_types[:, 0], self.cond_w_pdgids[:, 0]))
        b  = torch.cat((self.cond_bi_mask, self.cond_bi_types, self.cond_bi_pdgids))
        xo = (oh.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)  # (B, L, K * in_dim)
        return (xo @ w.transpose(1, 2).reshape(-1, self.h_dim) + oh @ b) / 3

    def _project_out(self, h: Tensor, mask: Tensor, oh: Tensor) -> Tensor:
        """Reads the velocity out of ``h``: ``v_i = mean_k(V_k[c_ik] h_i + a_k[c_ik])``, zero on conditioning tokens."""
        w   = torch.cat((self.cond_w_mask[:, 1], self.cond_w_types[:, 1], self.cond_w_pdgids[:, 1]))
        b   = torch.cat((self.cond_bo_mask, self.cond_bo_types, self.cond_bo_pdgids))
        out = (h @ w.permute(1, 0, 2).reshape(self.h_dim, -1)).view(*h.shape[:-1], -1, self.in_dim)
        out = torch.einsum("blki,blk->bli", out, oh) + oh @ b
        return (mask == 1).unsqueeze(-1) * out / 3

    def _time_emb(self, t: Tensor, d: Tensor | None) -> Tensor:
        """Computes ``emb(t, d) = sinusoid(t) + step_gain sin(d freqs_d)``; (..., h_dim) for ``t``, ``d`` (...)."""
        phase = t.unsqueeze(-1) * self.freqs
        emb   = torch.where(self.mask_freqs.bool(), torch.sin(phase), torch.cos(phase))
        if self.step_cond and d is not None:
            emb = emb + self.step_gain * torch.sin(d.unsqueeze(-1) * self.freqs_d)
        return emb

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
        t: Tensor | None = None,
        d: Tensor | None = None,
        z: Tensor | None = None,
    ) -> Tensor:
        """Predicts the velocity ``v`` of every token.

        Args:
            x:         (B, L, in_dim), the tokens.
            mask:      (B, L), 1 on generated tokens.
            attn_mask: (B, L) bool, False on padding.
            types:     (1, L), each position's type.
            pdgids:    (B, L, 1), each token's species index.
            t:         (B or 1, 1), the event's time.
            d:         (B or 1, 1), the event's step size; read only with ``step_cond``.
            z:         (B, lat_dim), the event's latent; only with ``lat_dim``.

        Returns:
            ``v`` (B, L, in_dim); with ``lat_dim`` ``(v, v_z)``, the latent's velocity (B, lat_dim).
        """
        if self._needs_legacy(t, d): return self._legacy_forward(x, mask, attn_mask, types, pdgids, t, d, z)
        B, L  = x.shape[:2]
        gen   = mask == 1
        c_con = self._time_emb(t.new_ones(1, 1), None if d is None else d.new_zeros(1, 1)).expand(B, 1, -1)
        c_gen = self._time_emb(t, d).expand(B, 1, -1)

        oh  = self._one_hot(mask, types, pdgids)
        emb = torch.where(gen.unsqueeze(-1), c_gen, c_con)
        h   = self._embed(x, oh) + emb
        seq = (h, emb, gen, attn_mask)
        if self.geom is not None:   # K conditioning tokens
            if self.geom_hide:
                seq = (h, emb, gen, attn_mask.index_fill(1, self.geom_scalar_tokens, False))
            seq = _prepend(self.geom().to(h.dtype).expand(B, -1, -1), c_con, False, *seq)
        if self.lat_dim:            # one generated token
            seq = _prepend(self.lat_in(z.to(h.dtype)).unsqueeze(1), c_gen, True, *seq)
        h, emb, gen, attn_mask = seq

        h = self.vf.attn_layers(self.vf.emb_dropout(h), mask=attn_mask,
                                condition=per_event_condition(emb, c_con, c_gen, gen) if self._per_event_cond else emb)
        v = self._project_out(h[:, -L:], mask, oh)
        return (v, self.lat_out(h[:, 0])) if self.lat_dim else v
