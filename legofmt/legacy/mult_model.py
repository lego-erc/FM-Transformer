"""Count-model paths the recipe no longer takes, kept so older configs and checkpoints still run unchanged:

    InvModel        ``mm_conf.train_inverse``: the incoming species predicted from the outgoing set, which backs
                    ``GenerateIn`` (``inverse_tensors``: its dataset tensors, ``inverse_pid``: its inference)
    ResidualHead    ``passthrough_depth > 0``: the pass-through head as residual blocks
    in_embd head    ``passthrough_on: in_embd`` (the config default): the pass-through head reads the incoming
                    embedding instead of the trunk's first output; one inline branch in ``MultModel``

``MultModel`` reaches each in one line. A config without ``opt_conf`` is ``legacy.optimizers.default_schedulefree``.
"""

import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder

from legofmt.compat import fp32_attention, needs_fp32_attention
from legofmt.geometry.geom_trafos import GeomTrafos


class ResidualHead(nn.Module):
    """depth x pre-norm [Linear -> Mish -> Linear] residual blocks, then a
    linear read-out; the shape of the trunk's own slot-0 computation."""

    def __init__(self, h: int, d: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.RMSNorm(h), nn.Linear(h, d), nn.Mish(), nn.Linear(d, h))
            for _ in range(depth)
        ])
        self.out = nn.Sequential(nn.RMSNorm(h), nn.Linear(h, 1))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return self.out(x)


class InvModel(nn.Module):
    """Inverse-direction predictor: outgoing shower (set) -> incoming PID."""

    def __init__(self, rc):
        super().__init__()
        self.rc = rc
        self.geom_trafos = GeomTrafos()
        self.register_buffer("pos_scale", rc.pos_scale, persistent=False)

        self.model = ContinuousTransformerWrapper(
            max_seq_len=rc.mm_conf.get("max_out_particles", 0) + 1,
            emb_dropout=rc.dropout,
            use_abs_pos_emb=rc.use_abs_pos_emb,
            post_emb_norm=rc.post_emb_norm,
            attn_layers=Encoder(
                dim=rc.inv_h_dim,
                depth=rc.inv_n_layers,
                heads=rc.inv_n_heads,
                attn_dropout=rc.dropout,
                ff_dropout=rc.dropout,
                dim_condition=rc.inv_h_dim,
                **rc.inv_model_args,
            ),
        )
        if needs_fp32_attention(rc.inv_model_args):
            fp32_attention(self.model)

        self.proj_in_ = nn.Linear(rc.in_dim, rc.inv_h_dim)
        self.embd_out_ = nn.Embedding(rc.ptypes.shape[0], rc.inv_h_dim)
        self.proj_cond_ = nn.Linear(1, rc.inv_h_dim)
        self.embd_query_ = nn.Parameter(torch.randn(rc.inv_h_dim) * 0.02)
        self.proj_pid_ = nn.Linear(rc.inv_h_dim, rc.n_ptypes_in)

    def proj_in(self, x):
        return self.proj_in_(torch.cat((x[..., :-6], self.geom_trafos.to_cuboid(x[..., -6:], d=self.pos_scale)), -1))

    def forward(self, out_tok, out_pid_idx, out_mask, edep):
        """Predicts the incoming-PID logits from the outgoing set's tokens, pids, mask and E_dep."""
        tok = self.proj_in(out_tok) + self.embd_out_(out_pid_idx)
        out = self.model(
            tok, mask=out_mask, condition=self.proj_cond_(edep.unsqueeze(-1)),
            prepend_embeds=self.embd_query_.expand(tok.shape[0], 1, -1),
            prepend_mask=out_mask.new_ones(tok.shape[0], 1),
        )
        return self.proj_pid_(out[:, 0])


def inverse_tensors(lds, conds: torch.Tensor, ptypes: torch.Tensor) -> tuple:
    """Returns ``InvModel``'s dataset tensors ``(out_tok, out_pid_idx, out_mask, edep)``.

    It needs every outgoing token, so it materialises the padded tensors.
    """
    ds = lds.data
    ds_f = ds.f
    out_cc = ds_f.out_cc.nan_to_num()
    out_tok = torch.cat((conds.unsqueeze(1).expand(-1, out_cc.shape[1], -1), out_cc), dim=-1).contiguous()
    out_pid_idx = torch.searchsorted(ptypes, ds_f.out_p[..., -1].long()).clamp(max=ptypes.shape[0] - 1)
    return out_tok, out_pid_idx, ds.am.out_p.bool().clone(), ds_f.edep.clone()


def inverse_pid(model, batch: tuple) -> torch.Tensor:
    """Returns the incoming species index ``InvModel`` picks for ``(out_tok, out_pid_idx, out_mask, edep)``."""
    if model.inv is None:
        raise RuntimeError(
            "MultModel was built with train_inverse=False; inverse-PID "
            "inference is unavailable. Retrain with mm_conf.train_inverse=true."
        )
    return model.inv(*batch[:4]).argmax(-1)
