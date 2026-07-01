import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.struct import cond_scalars
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.mod_comps.config import resolve_mult_config
from legofmt.mod_comps.optimizers import build_optimizer, schedulefree_adamw


class MultLoader(torch.utils.data.Dataset):
    def __init__(self, config: dict, device: str = "cpu", path: str = None):
        self.device = device
        mm_conf = config.get("mm_conf")
        lds_conf = config.get("dl_conf").get("lds_args").copy()
        if path is not None:
            lds_conf["path"] = path
        max_particles = mm_conf.get("max_out_particles")
        ptypes = mm_conf.get("ptypes", torch.tensor([11, 22]))
        ptypes_in = mm_conf.get("ptypes_in", torch.tensor([11, 22]))
        self.train_inverse = mm_conf.get("train_inverse", False)

        ds = LEGODataset(**lds_conf).data
        ds_f = ds.f
        # in_dim must be len(cond_scalars) + 7
        conds = torch.cat([ds_f.cond(n).unsqueeze(-1) for n in cond_scalars()], dim=-1)

        self.pdgid_in_idx = torch.searchsorted(
            ptypes_in, ds_f.in_p[..., 0, -1].contiguous()
        ).clamp(0, ptypes_in.shape[0] - 1)
        self.in_tok = torch.cat((conds, ds_f.in_cc.squeeze(-2)), dim=-1)
        self.counts = (ds_f.out_p[..., -1:] == ptypes.view(1, 1, -1)).sum(1).clamp_max(max_particles - 1)

        if self.train_inverse:
            out_cc = ds_f.out_cc.nan_to_num()
            self.out_tok = torch.cat(
                (conds.unsqueeze(1).expand(-1, out_cc.shape[1], -1), out_cc), dim=-1
            ).contiguous()
            self.out_pid_idx = torch.searchsorted(ptypes, ds_f.out_p[..., -1].long()).clamp(max=ptypes.shape[0] - 1)
            self.out_mask = ds.am.out_p.bool()
            self.edep = ds_f.edep

    def __len__(self):
        return self.in_tok.shape[0]

    def __getitem__(self, idx):
        base = (
            self.in_tok[idx].to(self.device),
            self.counts[idx].to(self.device),
            self.pdgid_in_idx[idx].to(self.device),
        )
        if not self.train_inverse:
            return base
        return base + (
            self.out_tok[idx].to(self.device),
            self.out_pid_idx[idx].to(self.device),
            self.out_mask[idx].to(self.device),
            self.edep[idx].to(self.device),
        )


class InvModel(nn.Module):
    """Inverse-direction predictor: outgoing shower (set) -> incoming PID.

    A bidirectional :class:`x_transformers.Encoder` over the (padded,
    order-invariant) set of outgoing particles plus a prepended CLS/query
    token; the query position's output is projected to logits over the
    incoming-particle vocabulary. The scalar deposited energy conditions the
    adaptive norms. Fully self-contained -- it shares NO parameters with the
    count model, so co-training cannot corrupt the count task.
    """

    def __init__(self, rc):
        super().__init__()
        self.rc = rc
        self.geom_trafos = GeomTrafos()

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

        self.proj_in_ = nn.Linear(rc.in_dim, rc.inv_h_dim)
        self.embd_out_ = nn.Embedding(rc.ptypes.shape[0], rc.inv_h_dim)
        self.proj_cond_ = nn.Linear(1, rc.inv_h_dim)
        self.embd_query_ = nn.Parameter(torch.randn(rc.inv_h_dim) * 0.02)
        self.proj_pid_ = nn.Linear(rc.inv_h_dim, rc.n_ptypes_in)

    def proj_in(self, x):
        x = x.clone()
        x[..., -6:] = self.geom_trafos.to_cube(x[..., -6:], d=self.rc.pos_scale)
        return self.proj_in_(x)

    def forward(self, out_tok, out_pid_idx, out_mask, edep):
        tok = self.proj_in(out_tok) + self.embd_out_(out_pid_idx)
        out = self.model(
            tok, mask=out_mask, condition=self.proj_cond_(edep.unsqueeze(-1)),
            prepend_embeds=self.embd_query_.expand(tok.shape[0], 1, -1),
            prepend_mask=out_mask.new_ones(tok.shape[0], 1),
        )
        return self.proj_pid_(out[:, 0])


class MultModel(LightningModule):
    def __init__(self, full_config: dict):
        super().__init__()
        rc = resolve_mult_config(full_config)
        self.rc = rc

        self.register_buffer("ptypes", rc.ptypes)
        self.register_buffer("ptypes_in", rc.ptypes_in)
        self.geom_trafos = GeomTrafos()

        self.model = ContinuousTransformerWrapper(
            max_seq_len=rc.max_seq_len,
            emb_dropout=rc.dropout,
            use_abs_pos_emb=rc.use_abs_pos_emb,
            post_emb_norm=rc.post_emb_norm,
            attn_layers=Decoder(
                dim=rc.h_dim,
                depth=rc.n_layers,
                heads=rc.n_heads,
                attn_dropout=rc.dropout,
                ff_dropout=rc.dropout,
                dim_condition=rc.h_dim,
                **rc.model_args,
            ),
        )

        self.proj_in_ = torch.nn.Linear(rc.in_dim, rc.h_dim)

        self.embd_pp_ = torch.nn.Embedding(rc.n_ptypes_in, rc.h_dim)
        self.embd_in_ = torch.nn.Embedding((rc.max_seq_len - 1) * rc.max_particles, rc.h_dim)
        self.register_buffer(
            "_in_offsets",
            torch.arange(rc.max_seq_len - 1, dtype=torch.long) * rc.max_particles,
            persistent=False,
        )
        self.proj_out_w = torch.nn.Parameter(torch.empty(rc.max_seq_len, rc.h_dim, rc.max_particles))
        self.proj_out_b = torch.nn.Parameter(torch.empty(rc.max_seq_len, rc.max_particles))
        _bound = 1.0 / (rc.h_dim ** 0.5)
        for _i in range(rc.max_seq_len):
            torch.nn.init.kaiming_uniform_(self.proj_out_w[_i], a=5 ** 0.5)
            torch.nn.init.uniform_(self.proj_out_b[_i], -_bound, _bound)

        # Optional inverse-PID co-training head (separate params, no sharing).
        # Built at construction time and, when present, trained on every step,
        # so DDP sees no unused parameters in either mode (plain ``ddp`` works).
        self.inv = InvModel(rc) if rc.train_inverse else None

        if rc.state_dict is not None:
            self.load_state_dict(rc.state_dict, strict=False)

        if rc.opt_conf is None:
            self.opt = schedulefree_adamw(
                self.parameters(),
                lr=rc.mm_conf.get("lr", 1e-3),
                betas=(0.95, 0.999),
                weight_decay=rc.mm_conf.get("weight_decay", 0.0),
                warmup_steps=rc.mm_conf.get("warmup_steps", 0),
            )
            self._sched = None
        else:
            self.opt, self._sched = build_optimizer(self.parameters(), rc.opt_conf)
        self._opt_is_sf = hasattr(self.opt, "train") and callable(getattr(self.opt, "train", None))

    def proj_in(self, x):
        x = x.clone()
        x[..., -6:] = self.geom_trafos.to_cube(x[..., -6:], d=self.rc.pos_scale)
        return self.proj_in_(x)

    def _opt_train(self):
        if self._opt_is_sf:
            self.opt.train()

    def _opt_eval(self):
        if self._opt_is_sf:
            self.opt.eval()

    def on_fit_start(self):
        self._opt_train()

    def on_fit_end(self):
        self._opt_eval()

    def training_step(self, batch, batch_idx):
        in_tok, counts, pdgid_in_idx = batch[:3]

        in_embd = self.proj_in(in_tok) + self.embd_pp_(pdgid_in_idx)
        in_seq = torch.cat(
            (in_embd.unsqueeze(1), self.embd_in_(counts[:, : self.rc.max_seq_len - 1] + self._in_offsets)),
            dim=1,
        )
        out_f = self.model(in_seq, mask=None, condition=in_embd)
        logits_f = torch.einsum("bsh,shp->bsp", out_f, self.proj_out_w) + self.proj_out_b
        loss_count = F.cross_entropy(logits_f.reshape(-1, self.rc.max_particles), counts.reshape(-1))

        if self.inv is None:
            self.log_dict(
                {"train_loss": loss_count, "loss/counts": loss_count.detach()},
                prog_bar=True, sync_dist=True,
            )
            return loss_count

        out_tok, out_pid_idx, out_mask, edep = batch[3:7]
        loss_inv = F.cross_entropy(self.inv(out_tok, out_pid_idx, out_mask, edep), pdgid_in_idx)
        loss = loss_count + loss_inv
        self.log_dict(
            {"train_loss": loss, "loss/counts": loss_count.detach(), "loss/pid_in": loss_inv.detach()},
            prog_bar=True, sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        if self._sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._sched}

    def train_dataloader(self):
        dataset_train = MultLoader(self.rc.config)
        num_workers = self.rc.dl_conf.get("num_workers", 4)
        return DataLoader(
            dataset_train,
            batch_size=self.rc.mm_conf.get("bs", 2**12),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    @torch.no_grad()
    def forward(self, batch: (tuple | torch.Tensor)):
        self._opt_eval()
        self.eval()
        if batch[0].dim() == 3:
            if self.inv is None:
                raise RuntimeError(
                    "MultModel was built with train_inverse=False; inverse-PID "
                    "inference is unavailable. Retrain with mm_conf.train_inverse=true."
                )
            return self.inv(*batch[:4]).argmax(-1)

        in_tok, _, pdgid_in_idx = batch
        in_embd = self.proj_in(in_tok) + self.embd_pp_(pdgid_in_idx)
        x = in_embd.unsqueeze(1)
        counts = in_tok.new_empty(in_tok.shape[0], self.rc.max_seq_len, dtype=torch.long)

        cache = None
        for i in range(self.rc.max_seq_len):
            out, cache = self.model(
                x,
                mask=None,
                condition=in_embd,
                return_intermediates=True,
                cache=cache,
                input_not_include_cache=(i > 0),
            )
            logits = out[:, -1] @ self.proj_out_w[i] + self.proj_out_b[i]
            sampled = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            counts[:, i] = sampled
            if i < self.rc.max_seq_len - 1:
                x = self.embd_in_(sampled + i * self.rc.max_particles).unsqueeze(1)
        return counts
