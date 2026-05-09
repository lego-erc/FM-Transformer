import torch
from torch import Tensor, nn
from x_transformers import ContinuousTransformerWrapper, Encoder


class CFMTrafo_x(nn.Module):
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
            # dim_in=h_dim,
            # dim_out=h_dim,
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
        n = states_mask.shape[1]
        key = None if self.training else (id(mask), id(types), id(pdgids), n)
        cache = getattr(self, "_inf_cache", None)
        if cache and cache[0] == key:
            _, b, bo, w_in, w_out, mask_eq1 = cache
        else:
            mi, ti, pi = mask.view(-1), types.view(-1)[:n], pdgids.view(-1)
            s3 = (-1, n, self.h_dim)
            so = (-1, n, self.in_dim)
            s4 = (-1, n, self.h_dim, self.in_dim)
            b = (self.b_mask_[mi].view(s3) + self.b_types_[ti].view(s3) + self.b_pdgids_[pi].view(s3)) / 3
            bo = (self.bo_mask_[mi].view(so) + self.bo_types_[ti].view(so) + self.bo_pdgids_[pi].view(so)) / 3
            w_in = self.l_mask_[mi, 0].view(s4) + self.l_types_[ti, 0] + self.l_pdgids_[pi, 0].view(s4)
            w_out = self.l_mask_[mi, 1].view(s4) + self.l_types_[ti, 1] + self.l_pdgids_[pi, 1].view(s4)
            mask_eq1 = mask == 1
            if key is not None:
                self._inf_cache = (key, b, bo, w_in, w_out, mask_eq1)

        tf = t.unsqueeze(-1) * self.freqs
        embd_t = torch.where(self.mask_freqs.bool(), tf.sin(), tf.cos())
        embd = torch.einsum("ijl, ijkl -> ijk", states_mask, w_in) / 3 + b + embd_t
        if self.training:
            embd = self.vf.emb_dropout(embd)
        x = self.vf.attn_layers(embd, mask=attn_mask, condition=embd_t)
        return mask_eq1 * (torch.einsum("ijk, ijkl -> ijl", x, w_out) / 3 + bo)