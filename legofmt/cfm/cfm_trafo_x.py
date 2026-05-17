"""Time-conditioned transformer with a factorized conditional projection.

Data flow per forward pass::

    x  ->  factorized up-projection    ->  project_in  ->  Encoder
                                                             |
    y  <-  factorized down-projection  <-  project_out  <----+

``project_in`` / ``project_out`` and the encoder come from
``x_transformers.ContinuousTransformerWrapper``; the factorized up- and
down-projections are this module.

Three discrete conditioning indices select the projection per token,
each of shape ``(B, n)``:

- ``mask``   (``nvtypes`` values) - ``0`` for conditioning slots, ``1``
  for slots the flow generates. Output is zeroed where ``mask != 1``.
- ``types``  (``ntypes`` values) - positional slot-type id; replaces
  absolute positional embeddings.
- ``pdgids`` (``npdgids`` values) - particle-species index.

Each source ``c`` owns:

- ``cond_w_{c}``  ``(card_c, 2, h_dim, in_dim)`` - ``[:, 0]`` is the
  up-projection, ``[:, 1]`` the down-projection.
- ``cond_bi_{c}`` ``(card_c, h_dim)``  - up-projection bias.
- ``cond_bo_{c}`` ``(card_c, in_dim)`` - down-projection bias.

Per-token up-projection (mean of three index-selected affine maps plus a
sinusoidal time embedding)::

    h_i = ( cond_w_mask  [mask  [i], 0] @ x_i + cond_bi_mask  [mask  [i]]
          + cond_w_types [types [i], 0] @ x_i + cond_bi_types [types [i]]
          + cond_w_pdgids[pdgids[i], 0] @ x_i + cond_bi_pdgids[pdgids[i]]
          ) / 3 + sincos_embed(t)[i]

Weights are summed before the einsum to fuse the contraction. The
down-projection mirrors this with ``cond_w_*[:, 1]`` and ``cond_bo_*``,
gated to ``mask == 1``.

Note:
    Legacy checkpoints use trailing-underscore names (``l_mask_``,
    ``b_mask_``, ...). A load pre-hook remaps the keys and emits one
    ``DeprecationWarning``.
"""
import torch
from torch import Tensor, nn
from x_transformers import ContinuousTransformerWrapper, Encoder


class CFMTrafo_x(nn.Module):
    """Conditional flow-matching transformer with factorized projections.

    See the module docstring for the projection construction and the
    meaning of ``mask`` / ``types`` / ``pdgids``.

    Args:
        h_dim: Encoder hidden dimension. Default: ``256``.
        nhead: Attention heads per encoder layer. Default: ``8``.
        max_seq_l: Maximum padded sequence length. Default: ``9``.
        nvtypes: Cardinality of ``mask`` (typically ``2``: conditioning
            vs. generated). Default: ``3``.
        ntypes: Cardinality of ``types``. Defaults to ``max_seq_l``.
        in_dim: Per-token feature dimension (same on the way in and out).
            Default: ``6``.
        ff_mult: Encoder feed-forward expansion ratio. Default: ``1``.
        dropout: Dropout on embeddings, attention, and feed-forward.
            Default: ``0.1``.
        nlayers: Number of encoder layers. Default: ``4``.
        xavier_gain: Gain for ``nn.init.xavier_normal_``. Default: ``1.0``.
        npdgids: Cardinality of ``pdgids`` (particle-species lookup).
            Default: ``1``.
        dim_in_out: When not ``None``, the wrapped
            ``ContinuousTransformerWrapper`` uses
            ``Linear(dim_in_out, h_dim)`` / ``Linear(h_dim, dim_in_out)``
            at its boundary instead of identities. Back-compat for
            checkpoints whose state dict contains ``vf.project_in.*`` /
            ``vf.project_out.*`` weights;
            ``legofmt.main.config._apply_legacy_projection_in_out`` sets
            this to ``h_dim`` when those keys are detected. Default:
            ``None``.
        **kwargs: Forwarded to ``x_transformers.Encoder`` (e.g.
            ``use_adaptive_rmsnorm``, ``attn_qk_norm``).
    """

    def __init__(
        self,
        h_dim: int,
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
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.max_seq_l = max_seq_l
        self.nvtypes = nvtypes
        self.ntypes = ntypes if ntypes is not None else max_seq_l
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

        self.l_mask_ = nn.Parameter(torch.empty(self.nvtypes, 2, self.h_dim, self.in_dim))
        self.b_mask_ = nn.Parameter(torch.empty(self.nvtypes, self.h_dim))
        self.bo_mask_ = nn.Parameter(torch.empty(self.nvtypes, self.in_dim))
        self.l_types_ = nn.Parameter(torch.empty(self.ntypes, 2, self.h_dim, self.in_dim))
        self.b_types_ = nn.Parameter(torch.empty(self.ntypes, self.h_dim))
        self.bo_types_ = nn.Parameter(torch.empty(self.ntypes, self.in_dim))
        self.l_pdgids_ = nn.Parameter(torch.empty(self.npdgids, 2, self.h_dim, self.in_dim))
        self.b_pdgids_ = nn.Parameter(torch.empty(self.npdgids, self.h_dim))
        self.bo_pdgids_ = nn.Parameter(torch.empty(self.npdgids, self.in_dim))

        for p in (self.l_mask_, self.b_mask_, self.bo_mask_,
            self.l_types_, self.b_types_, self.bo_types_,
            self.l_pdgids_, self.b_pdgids_, self.bo_pdgids_):
            nn.init.xavier_normal_(p, gain=xavier_gain)

        self.freqs = nn.Parameter(
            self.h_dim * 1e-4 ** (torch.arange(self.h_dim) / self.h_dim),
            requires_grad=False,
        )
        self.register_buffer("mask_freqs", torch.arange(self.h_dim) % 2)

    def forward(
        self,
        t: Tensor,
        states_mask: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
    ) -> Tensor:
        """Compute the velocity field at flow time ``t``.

        Args:
            t: Flow time, broadcastable to ``(B, 1)``.
            x: Per-token features, shape ``(B, n, in_dim)``.
            mask: Token-role ids, shape ``(B, n)``, values in
                ``[0, nvtypes)``. The output is zeroed where ``mask != 1``.
            attn_mask: Boolean attention mask, shape ``(B, n)``.
            types: Slot-type ids, shape ``(B, n)``, values in
                ``[0, ntypes)``.
            pdgids: Particle-species indices, shape ``(B, n)``, values in
                ``[0, npdgids)``.

        Returns:
            Velocity field, shape ``(B, n, in_dim)``, zeroed where
            ``mask != 1``.
        """
        mi, ti, pi = mask.view(-1), types.view(-1)[:n], pdgids.view(-1)
        s3 = (-1, n, self.h_dim)
        so = (-1, n, self.in_dim)
        s4 = (-1, n, self.h_dim, self.in_dim)

        tf = t.unsqueeze(-1) * self.freqs
        embd_t = torch.where(self.mask_freqs.bool(), tf.sin(), tf.cos())
        embd = ((torch.einsum("ijl, ijkl -> ijk", states_mask,
            self.l_mask_[mi, 0].view(s4) + self.l_types_[ti, 0] + self.l_pdgids_[pi, 0].view(s4))
            + (self.b_mask_[mi].view(s3) + self.b_types_[ti].view(s3) + self.b_pdgids_[pi].view(s3))) / 3
            + embd_t)
        embd = self.vf.project_in(embd)
        if self.training:
            embd = self.vf.emb_dropout(embd)
        x = self.vf.attn_layers(embd, mask=attn_mask, condition=embd_t)
        x = self.vf.project_out(x)
        return (mask == 1).unsqueeze(-1) * (torch.einsum("ijk, ijkl -> ijl", x,
            self.l_mask_[mi, 1].view(s4) + self.l_types_[ti, 1] + self.l_pdgids_[pi, 1].view(s4))
            + self.bo_mask_[mi].view(so) + self.bo_types_[ti].view(so) + self.bo_pdgids_[pi].view(so)) / 3
