"""The flow's velocity field: a transformer over the event's padded token sequence.

```
tokens     [z | g_1 .. g_K | c | x_1 .. x_L]           x: the event's tokens; optional: z, the event's latent,
                                                       g, the geometry tokens, and c, the pooled conditioning scalars

embed      h_i = mean_k(W_k[c_ik] x_i + b_k[c_ik])     c_ik: token i's mask value, type and species index (k = 1, 2, 3);
                                                       W_k and b_k hold one linear map per value
           h_z = lat_in z,   h_g = GeomEncoder(),   h_c = CondPool(conditioning scalars)

time       h_i = h_i + emb_i,   emb_i = emb(t, d) on a generated token, emb(1, 0) on a conditioning one
           emb(t, d) = sinusoid(t) + step_gain sin(d freqs_d)

encoder    h = Encoder(h | emb)                         every block's adaptive RMSNorm / layer scale reads emb_i

read-out   v_i = mean_k(V_k[c_ik] h_i + a_k[c_ik])      on a generated token, 0 on a conditioning one
           v_z = lat_out h_z
```

A generated token (``mask == 1``) is one the flow moves: the outgoing tokens, the E_dep token, the latent. A
conditioning token holds data and is not moved, so its time embedding is that of ``t = 1, d = 0``: the conditioning
scalar tokens, the incoming token, the geometry tokens, ``c``. ``(t, d)`` is one time and one step size per event.

Four parts are optional:

    step_cond      the step size ``d`` of a flow-map step enters the time embedding
    geom_encoder   K summary tokens of a layered stack (``geom_encoder.py``), prepended as conditioning tokens
    lat_dim        the event's latent ``z`` (``model_conf.edep_ae``), prepended as one generated token
    cond_pool      the conditioning scalars as one token ``c`` (``cond_pool.py``); their own tokens leave the sequence.
                   Not together with ``geom_encoder`` (``config._check_lego``)

The encoder is an x-transformers ``Encoder``. The options ``time_cond: false``, per-token times, ``dim_in_out`` and
``grad_ckpt`` are legacy (:mod:`legofmt.legacy.cfm_trafo_x`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from x_transformers import ContinuousTransformerWrapper, Encoder

from legofmt.cfm.cond_pool import CondPool
from legofmt.cfm.geom_encoder import GeomEncoder
from legofmt.cfm.event_cond import per_event_condition, per_event_gammas
from legofmt.compat import fp32_attention, legacy_param_rename, needs_fp32_attention
from legofmt.legacy.cfm_trafo_x import LegacyField


class CFMTrafo_x(LegacyField):
    """The flow's velocity field: a transformer over the event's tokens (module docstring).

    Its arguments are the config's ``model_conf.model_args`` (README); those it does not name go to the
    x-transformers ``Encoder``.
    """

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
        cond_pool: int = 0,
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

        self._add_geometry_tokens(geom_encoder, h_dim)   # ``model_args.geom_encoder``: ``self.geom``, None without one
        # ``model_args.cond_pool``: the number of prefix tokens (the config sets it from ``true``), 0 without pooling.
        # E_dep is token 1; every other prefix token holds one conditioning scalar
        self.n_prefix  = cond_pool
        self.cond_pool = CondPool(cond_pool - 1, h_dim) if cond_pool else None

        # the encoder
        self.vf = ContinuousTransformerWrapper(
            dim_in=dim_in_out,       # legacy; None: no linear map into and out of the encoder
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
        self._per_event_cond = self._patch_encoder(kwargs, time_cond)   # fp32 attention, per-event gammas

        # token embedding and read-out: for every value of the mask, the type and the pdgid, one linear map into the
        # encoder (``cond_w_*[:, 0]``: W, ``cond_bi_*``: b) and one out of it (``cond_w_*[:, 1]``: V, ``cond_bo_*``: a)
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

        # the latent token (``model_conf.edep_ae``); its read-out starts at zero, so an untrained latent has no velocity
        if lat_dim:
            self.lat_in  = nn.Linear(lat_dim, h_dim)
            self.lat_out = nn.Linear(h_dim, lat_dim)
            nn.init.zeros_(self.lat_out.weight)
            nn.init.zeros_(self.lat_out.bias)

        # Xavier's std times 3 ** 0.5: the mean of the three maps then has Xavier's std
        w_std = xavier_gain * 3 ** 0.5 * (2.0 / (in_dim + h_dim)) ** 0.5
        for p in (self.cond_w_mask, self.cond_w_types, self.cond_w_pdgids):
            nn.init.normal_(p, std=w_std)
        for p in (
            self.cond_bi_mask, self.cond_bi_types, self.cond_bi_pdgids,
            self.cond_bo_mask, self.cond_bo_types, self.cond_bo_pdgids,
        ):
            nn.init.zeros_(p)

        # time embedding: sinusoids of t at frequencies from h_dim down to ~1e-4 h_dim; with ``step_cond`` a second set
        # for d, behind a gain that starts at zero
        self.register_buffer("freqs", h_dim * 1e-4 ** (torch.arange(h_dim) / h_dim))
        self.register_buffer("mask_freqs", torch.arange(h_dim) % 2)
        if step_cond:
            self.register_buffer("freqs_d", 2 * torch.pi * 2 ** (torch.arange(h_dim) * 3.0 / h_dim), persistent=False)
            self.step_gain = nn.Parameter(torch.zeros(1))

        self._init_legacy(time_cond, dim_in_out, grad_ckpt)

    def _embed(self, x: Tensor, oh: Tensor) -> Tensor:
        """Embeds the tokens ``x`` (B, L, in_dim): ``h_i = mean_k(W_k[c_ik] x_i + b_k[c_ik])``, (B, L, h_dim).

        ``oh`` is the tokens' one-hot (``_one_hot``). The sum is one matmul of the Kronecker product ``oh (x) x``
        with the stacked maps. Selecting a map per token instead builds a ``(B, L, h_dim, in_dim)`` intermediate,
        whose index backward is 60% of an eager step.
        """
        w  = torch.cat((self.cond_w_mask[:, 0], self.cond_w_types[:, 0], self.cond_w_pdgids[:, 0]))
        b  = torch.cat((self.cond_bi_mask, self.cond_bi_types, self.cond_bi_pdgids))
        xo = (oh.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)  # (B, L, (nvtypes + ntypes + npdgids) * in_dim)
        return (xo @ w.transpose(1, 2).flatten(0, 1) + oh @ b) / 3

    def _project_out(self, h: Tensor, mask: Tensor, oh: Tensor) -> Tensor:
        """Reads the velocity out of ``h``: ``v_i = mean_k(V_k[c_ik] h_i + a_k[c_ik])``, zero on conditioning tokens.

        ``h`` is (B, L, h_dim), ``mask`` (B, L) and ``oh`` the tokens' one-hot (``_one_hot``); returns (B, L, in_dim).
        """
        w   = torch.cat((self.cond_w_mask[:, 1], self.cond_w_types[:, 1], self.cond_w_pdgids[:, 1]))
        b   = torch.cat((self.cond_bo_mask, self.cond_bo_types, self.cond_bo_pdgids))
        out = (h @ w.transpose(0, 1).flatten(1)).unflatten(-1, (-1, self.in_dim))
        out = torch.einsum("blki,blk->bli", out, oh) + oh @ b
        return (mask == 1).unsqueeze(-1) * out / 3

    def _time_emb(self, t: Tensor, d: Tensor | None) -> Tensor:
        """Computes ``emb(t, d) = sinusoid(t) + step_gain sin(d freqs_d)``: (..., h_dim) for ``t`` and ``d`` (...).

        The ``d`` term is there only with ``step_cond`` and a ``d`` that is not None.
        """
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
            types:     (1, L), each position's type index.
            pdgids:    (B, L, 1), each token's species index.
            t:         (B or 1, 1), the event's time.
            d:         (B or 1, 1), the event's step size; read only with ``step_cond``.
            z:         (B, lat_dim), the event's latent; only with ``lat_dim``.

        Returns:
            ``v`` (B, L, in_dim), zero on conditioning tokens; with ``lat_dim`` the pair ``(v, v_z)``, ``v_z``
            (B, lat_dim) the latent's velocity.
        """
        if self._needs_legacy(t, d): return self._legacy_forward(x, mask, attn_mask, types, pdgids, t, d, z)  # legacy
        gen          = mask == 1
        c_con, c_gen = self._time_conditions(t, d, len(x))     # emb(1, 0), emb(t, d)
        oh           = self._one_hot(mask, types, pdgids)
        emb          = torch.where(gen.unsqueeze(-1), c_gen, c_con)
        h            = self._embed(x, oh) + emb
        h, emb, gen, attn_mask = self._with_extra_tokens(h, emb, gen, attn_mask, c_con, c_gen, z, x)   # [z | g | c | x]
        h            = self._encode(h, emb, gen, attn_mask, c_con, c_gen)
        v            = self._project_out(self._token_states(h, x.shape[1]), mask, oh)
        return (v, self.lat_out(h[:, 0])) if self.lat_dim else v

    # ---- plumbing -----------------------------------------------------------------------------------------------

    def _add_geometry_tokens(self, geom_encoder: dict | None, h_dim: int) -> None:
        """Builds ``self.geom``, the encoder of the geometry tokens (``model_args.geom_encoder``); None without one.

        With ``hide_scalars`` the geometry tokens stand in for the conditioning scalar tokens, which are masked out of
        the attention: ``geom_scalar_tokens``, the prefix tokens except the E_dep token (token 1).
        """
        self.geom = None
        if geom_encoder is not None:
            g = dict(geom_encoder)
            n_prefix, self.geom_hide = g.pop("n_prefix"), g.pop("hide_scalars")
            self.register_buffer("geom_scalar_tokens", torch.tensor([0, *range(2, n_prefix)]), persistent=False)
            self.geom = GeomEncoder(g.pop("layers"), h_dim, **g)

    def _patch_encoder(self, kwargs: dict, time_cond: bool) -> bool:
        """Patches the encoder: fp32 attention where the options ``kwargs`` need it, and the norms' per-event condition.

        Returns:
            Whether the norms read the per-event condition (``event_cond.py``); never without ``time_cond``.
        """
        if needs_fp32_attention(kwargs):
            fp32_attention(self.vf)
        return time_cond and per_event_gammas(self.vf.attn_layers)

    def _one_hot(self, mask: Tensor, types: Tensor, pdgids: Tensor) -> Tensor:
        """Builds each token's one-hot over the three vocabularies: its mask value, its type, its species index.

        ``_embed`` and ``_project_out`` select a token's linear maps by one matmul with it.

        Args:
            mask:   (B, L), 1 on generated tokens.
            types:  (1, L), each position's type index.
            pdgids: (B, L, 1), each token's species index (``LEGOLtng.species_idx``); 0: unknown or padding.

        Returns:
            (B, L, nvtypes + ntypes + npdgids).
        """
        B, L = mask.shape[:2]
        return torch.cat((
            F.one_hot(mask.reshape(B, L).long(), self.nvtypes),
            F.one_hot(types.flatten()[:L], self.ntypes).expand(B, -1, -1),
            F.one_hot(pdgids.reshape(B, L).long(), self.npdgids),
        ), dim=-1).to(self.cond_w_mask.dtype)

    def _time_conditions(self, t: Tensor, d: Tensor | None, B: int) -> tuple[Tensor, Tensor]:
        """Returns ``(c_con, c_gen)``: ``emb(1, 0)`` for conditioning tokens, ``emb(t, d)`` for generated ones.

        Each is (B, 1, h_dim).
        """
        c_con = self._time_emb(t.new_ones(1, 1), None if d is None else d.new_zeros(1, 1)).expand(B, 1, -1)
        c_gen = self._time_emb(t, d).expand(B, 1, -1)
        return c_con, c_gen

    def _with_extra_tokens(self, h: Tensor, emb: Tensor, gen: Tensor, attn_mask: Tensor, c_con: Tensor,
                           c_gen: Tensor, z: Tensor | None, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepends the field's optional tokens to the sequence: ``[z | g | c | x]``.

        The pooled token ``c`` and the geometry tokens ``g`` are conditioning tokens, the latent token ``z`` is a
        generated one. With ``cond_pool`` the conditioning scalar tokens leave the sequence; with ``geom_hide`` they
        stay and are masked out of the attention.

        ``(h, emb, gen, attn_mask)`` is the sequence of the event's L tokens: their embeddings and time embeddings
        (B, L, h_dim), the generated tokens and the attention mask (B, L). ``c_con`` and ``c_gen`` (B, 1, h_dim) are
        ``emb(1, 0)`` and ``emb(t, d)``, ``z`` is the event's latent (B, lat_dim) or None. ``x`` are the tokens
        (B, L, in_dim): channel 0 of a conditioning scalar token holds the scalar's value. Returns the four of the
        sequence with the optional tokens.
        """
        seq = (h, emb, gen, attn_mask)
        if self.cond_pool is not None:   # one conditioning token from the scalars' values, x[:, scalar tokens, 0]
            P, values = self.n_prefix, torch.cat((x[:, :1, 0], x[:, 2:self.n_prefix, 0]), dim=1)
            seq = _prepend(self.cond_pool(values).to(h.dtype), c_con, False,
                           *(torch.cat((s[:, 1:2], s[:, P:]), dim=1) for s in seq))
        if self.geom is not None:   # K conditioning tokens
            if self.geom_hide:
                seq = (h, emb, gen, attn_mask.index_fill(1, self.geom_scalar_tokens, False))
            seq = _prepend(self.geom().to(h.dtype).expand(len(h), -1, -1), c_con, False, *seq)
        if self.lat_dim:            # one generated token
            seq = _prepend(self.lat_in(z.to(h.dtype)).unsqueeze(1), c_gen, True, *seq)
        return seq

    def _token_states(self, h: Tensor, L: int) -> Tensor:
        """Returns the encoder's states (B, L, h_dim) of the event's own ``L`` tokens, without the prepended ones.

        With ``cond_pool`` the conditioning scalar tokens were not in the sequence: their states are zero. They are
        conditioning tokens, so ``_project_out`` reads no velocity from them.
        """
        if self.cond_pool is None:
            return h[:, -L:]
        P    = self.n_prefix
        own  = h[:, P - 1 - L:]   # the E_dep token, the incoming token, the outgoing tokens
        gone = h.new_zeros(len(h), P - 2, h.shape[-1])
        return torch.cat((gone[:, :1], own[:, :1], gone, own[:, 1:]), dim=1)

    def _encode(self, h: Tensor, emb: Tensor, gen: Tensor, attn_mask: Tensor, c_con: Tensor, c_gen: Tensor) -> Tensor:
        """Runs the encoder on ``h`` with the time condition ``emb``, in its per-event form where the norms read that.

        ``c_con``, ``c_gen`` and ``gen`` build the per-event form (``event_cond.per_event_condition``).
        """
        h         = self.vf.emb_dropout(h)
        condition = per_event_condition(emb, c_con, c_gen, gen) if self._per_event_cond else emb
        return self.vf.attn_layers(h, mask=attn_mask, condition=condition)


# ---- plumbing ------------------------------------------------------------------------------------------------------

def _prepend(tokens: Tensor, c: Tensor, generated: bool, h: Tensor, emb: Tensor, gen: Tensor, attn_mask: Tensor):
    """Prepends ``tokens`` (B, n, h_dim) to the sequence ``(h, emb, gen, attn_mask)`` and returns the longer one.

    The time embedding ``c`` (B, 1, h_dim) is added to them. They are attended to, and are generated tokens if
    ``generated``, conditioning tokens otherwise.
    """
    n = tokens.shape[1]
    c = c.expand(-1, n, -1)
    return (torch.cat((tokens + c, h), dim=1), torch.cat((c, emb), dim=1),
            F.pad(gen, (n, 0), value=generated), F.pad(attn_mask, (n, 0), value=True))
