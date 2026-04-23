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
        self.npdgids = npdgids
        self.vl = (2, self.h_dim, self.in_dim)
        self.vb = (self.h_dim,)
        self.vbo = (self.in_dim,)

        self.vf = ContinuousTransformerWrapper(
            dim_in=h_dim,
            dim_out=h_dim,
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
        self.l_types_ = nn.Parameter(torch.empty(self.max_seq_l, 2, self.h_dim, self.in_dim))
        self.b_types_ = nn.Parameter(torch.empty(self.max_seq_l, self.h_dim))
        self.bo_types_ = nn.Parameter(torch.empty(self.max_seq_l, self.in_dim))
        self.l_pdgids_ = nn.Parameter(torch.empty(self.npdgids, 2, self.h_dim, self.in_dim))
        self.b_pdgids_ = nn.Parameter(torch.empty(self.npdgids, self.h_dim))
        self.bo_pdgids_ = nn.Parameter(torch.empty(self.npdgids, self.in_dim))

        par_list = [
            self.l_mask_,
            self.b_mask_,
            self.bo_mask_,
            self.l_types_,
            self.b_types_,
            self.bo_types_,
            self.l_pdgids_,
            self.b_pdgids_,
            self.bo_pdgids_,
            ]
        
        for p in par_list:
            nn.init.xavier_normal_(p, gain=xavier_gain)

        self.freqs = nn.Parameter(
            self.h_dim * 1e-4 ** (torch.arange(self.h_dim) / self.h_dim),
            requires_grad=False,
        )
        self.mask_freqs = nn.Parameter(
            torch.remainder(torch.arange(self.h_dim), 2), requires_grad=False
        )
        self.mask_freqs_rolled = nn.Parameter(
            torch.remainder(torch.arange(self.h_dim) + 1, 2), requires_grad=False
        )

    def forward(
        self,
        t: Tensor,
        states_mask: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
    ) -> Tensor:
        n_tokens = states_mask.shape[1]
        mask_idx = mask.view(-1)
        types_idx = types.view(-1)[:n_tokens]
        pdgids_idx = pdgids.view(-1)

        b_embd = (
            self.b_mask_.index_select(0, mask_idx).view(-1, n_tokens, *self.vb)
            + self.b_types_.index_select(0, types_idx).view(-1, n_tokens, *self.vb)
            + self.b_pdgids_.index_select(0, pdgids_idx).view(-1, n_tokens, *self.vb)
        ) / 3
        bo_embd = (
            self.bo_mask_.index_select(0, mask_idx).view(-1, n_tokens, *self.vbo)
            + self.bo_types_.index_select(0, types_idx).view(-1, n_tokens, *self.vbo)
            + self.bo_pdgids_.index_select(0, pdgids_idx).view(-1, n_tokens, *self.vbo)
        ) / 3

        t_freqs = torch.einsum("ij, k -> ijk", t, self.freqs)
        embd_t = self.mask_freqs * t_freqs.sin() + self.mask_freqs_rolled * t_freqs.cos()

        l_embdd = 0
        for w_full, idx in (
            (self.l_mask_, mask_idx),
            (self.l_types_, types_idx),
            (self.l_pdgids_, pdgids_idx),
        ):
            w_in = w_full[:, 0].index_select(0, idx).view(-1, n_tokens, self.h_dim, self.in_dim)
            l_embdd = l_embdd + torch.einsum("ijl, ijkl -> ijk", states_mask, w_in)
        l_embdd = l_embdd / 3

        embdd = l_embdd + b_embd + embd_t

        x = self.vf.project_in(embdd)
        x = x + self.vf.pos_emb(x)
        x = self.vf.post_emb_norm(x)
        x = self.vf.emb_dropout(x)
        x = self.vf.attn_layers(x, mask=attn_mask, condition=embd_t)
        trafo_out = self.vf.project_out(x)

        l_out = 0
        for w_full, idx in (
            (self.l_mask_, mask_idx),
            (self.l_types_, types_idx),
            (self.l_pdgids_, pdgids_idx),
        ):
            w_out = w_full[:, 1].index_select(0, idx).view(-1, n_tokens, self.h_dim, self.in_dim)
            l_out = l_out + torch.einsum("ijk, ijkl -> ijl", trafo_out, w_out)
        l_out = l_out / 3

        out = l_out + bo_embd

        return (mask == 1) * out