import torch
from torch import Tensor, nn
from x_transformers import ContinuousTransformerWrapper, Encoder


class CFMTrafo_x(nn.Module):
    def __init__(
        self,
        h_dim: int,
        *,
        nhead: int = 8,
        ntokens: int = 4,
        nvtypes: int = 3,
        in_dim: int = 6,
        ff_mult: int = 1,
        dropout: float = 0.1,
        nlayers: int = 4,
        npdgids: int = 1,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.ntokens = ntokens
        self.nvtypes = nvtypes
        self.npdgids = npdgids
        self.vl = (self.ntokens, 2, self.h_dim, self.in_dim)
        self.vb = (self.ntokens, self.h_dim)
        self.vbo = (self.ntokens, self.in_dim)

        self.vf = ContinuousTransformerWrapper(
            max_seq_len=ntokens,
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

        self.l_mask_ = nn.Embedding(self.nvtypes, 2 * self.h_dim * self.in_dim)
        self.b_mask_ = nn.Embedding(self.nvtypes, self.h_dim)
        self.bo_mask_ = nn.Embedding(self.nvtypes, self.in_dim)
        self.l_types_ = nn.Embedding(self.ntokens, 2 * self.h_dim * self.in_dim)
        self.b_types_ = nn.Embedding(self.ntokens, self.h_dim)
        self.bo_types_ = nn.Embedding(self.ntokens, self.in_dim)
        self.l_pdgids_ = nn.Embedding(self.npdgids, 2 * self.h_dim * self.in_dim)
        self.b_pdgids_ = nn.Embedding(self.npdgids, self.h_dim)
        self.bo_pdgids_ = nn.Embedding(self.npdgids, self.in_dim)

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
        l_mask = self.l_mask_(mask.view(-1)).view(-1, *self.vl)
        b_mask = self.b_mask_(mask.view(-1)).view(-1, *self.vb)
        bo_mask = self.bo_mask_(mask.view(-1)).view(-1, *self.vbo)

        l_types = self.l_types_(types.view(-1)).view(-1, *self.vl)
        b_types = self.b_types_(types.view(-1)).view(-1, *self.vb)
        bo_types = self.bo_types_(types.view(-1)).view(-1, *self.vbo)

        l_pdgids = self.l_pdgids_(pdgids.view(-1)).view(-1, *self.vl)
        b_pdgids = self.b_pdgids_(pdgids.view(-1)).view(-1, *self.vb)
        bo_pdgids = self.bo_pdgids_(pdgids.view(-1)).view(-1, *self.vbo)

        t_freqs = torch.einsum("ij, k -> ijk", t, self.freqs)
        embd_t = self.mask_freqs * t_freqs.sin() + self.mask_freqs_rolled * t_freqs.cos()

        l_embd = 1/3 * (l_mask + l_types + l_pdgids)
        b_embd = 1/3 * (b_mask + b_types + b_pdgids)
        bo_embd = 1/3 * (bo_mask + bo_types + bo_pdgids)

        l_embdd = torch.einsum("ijl, ijkl -> ijk", states_mask, l_embd[:, :, 0])
        embdd = l_embdd + b_embd + embd_t

        trafo_out = self.vf(embdd, mask=attn_mask, condition=embd_t)
        
        l_out = torch.einsum("ijk, ijkl -> ijl", trafo_out, l_embd[:, :, 1])
        out = l_out + bo_embd

        return (mask == 1) * out