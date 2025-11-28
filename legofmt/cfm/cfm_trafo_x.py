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
        ninferences: int = 3,
        in_dim: int = 6,
        ff_mult: int = 1,
        dropout: float = 0.1,
        nlayers: int = 4,
        xavier_gain: float = 1.0,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.ntokens = ntokens
        self.ninferences = ninferences

        self.vf = ContinuousTransformerWrapper(
            dim_in=h_dim,
            dim_out=h_dim,
            max_seq_len=ntokens,
            emb_dropout=dropout,
            use_abs_pos_emb=False,
            attn_layers=Encoder(
                dim=h_dim,
                depth=nlayers,
                heads=nhead,
                rotary_pos_emb=False,
                layer_dropout=dropout,
                attn_dropout=dropout,
                ff_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                ff_mult=ff_mult,
                ff_no_bias=True,
                attn_flash=True,
                **kwargs,
            ),
        )

        self.embd_ = nn.Parameter(
            torch.empty(self.ninferences, self.ntokens, self.h_dim, self.in_dim)
        )
        self.bias_ = nn.Parameter(
            torch.zeros(self.ninferences, self.ntokens, self.h_dim)
        )
        nn.init.xavier_normal_(self.embd_, gain=xavier_gain)
        nn.init.xavier_normal_(self.bias_, gain=xavier_gain)

        self.lin_out_ = nn.Parameter(torch.empty(self.ntokens, self.h_dim, self.in_dim))
        self.bias_out_ = nn.Parameter(torch.zeros(self.ntokens, self.in_dim))

        self.freqs = nn.Parameter(
            self.h_dim * 1e-4 ** (torch.arange(self.h_dim) / self.h_dim), requires_grad=False,
        )
        self.mask_freqs = nn.Parameter(torch.remainder(torch.arange(128), 2), requires_grad=False)

        nn.init.xavier_normal_(self.lin_out_, gain=xavier_gain)
        nn.init.xavier_normal_(self.bias_out_, gain=xavier_gain)
        self.device = self.embd_.device

    def forward(
        self,
        t: Tensor,
        states_t: Tensor,
        mask: Tensor,
        attn_mask: (Tensor | None) = None,
        states_1: (Tensor | None) = None,
        types: (Tensor | None) = None,
        filtered: bool = True,
    ) -> Tensor:
        embd = self.embd_.gather(
            0, mask.view(*mask.shape[:2], 1, 1).expand(-1, -1, self.h_dim, self.in_dim)
        )
        bias = self.bias_.gather(
            0, mask.view(*mask.shape[:2], 1).expand(-1, -1, self.h_dim)
        )

        if types is not None:
            embd = embd.gather(
                1,
                types.view(*types.shape[:2], 1, 1).expand_as(embd),
            )
            bias = bias.gather(
                1,
                types.view(*types.shape[:2], 1).expand_as(bias),
            )
        if filtered:
            states_mask = states_t
        if not filtered:
            states_cat = torch.stack(
                (states_1, states_t, torch.ones_like(states_t)), dim=2
            )
            states_mask = states_cat.gather(
                2,
                mask.view(*mask.shape[:2], 1, 1).expand(-1, -1, 1, self.in_dim),
            ).squeeze(2)

        t_freqs = torch.einsum("ij, k -> ijk", t, self.freqs)
        embd_t = self.mask_freqs * t_freqs.sin() + (self.mask_freqs * t_freqs.cos()).roll(1, dims=-1)
        embd_type = torch.einsum("ijl, ijkl -> ijk", states_mask, embd)
        embdd = embd_type + bias + embd_t

        trafo_out = self.vf(embdd, mask=attn_mask)
        out_fwd = (
            torch.einsum("ijk, jkl -> ijl", trafo_out, self.lin_out_[: mask.shape[1]])
            + self.bias_out_[: mask.shape[1]]
        )

        return (mask == 1.0) * out_fwd
