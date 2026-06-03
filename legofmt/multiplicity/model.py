import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder

import schedulefree

from legofmt.data.dataloaders import LEGODataset
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.mod_comps.config import resolve_mult_config
from legofmt.mod_comps.optimizers import build_optimizer


class MultLoader(torch.utils.data.Dataset):
    def __init__(self, config: dict, device: str = "cpu", path: str = None):
        self.device = device
        mm_conf = config.get("mm_conf")
        lds_conf = config.get("dl_conf").get("lds_args").copy()
        if path is not None:
            lds_conf["path"] = path
        max_particles = mm_conf.get("max_out_particles")
        use_density = mm_conf.get("use_density", True)
        ptypes = mm_conf.get("ptypes", torch.tensor([11, 22]))
        ptypes_in = mm_conf.get("ptypes_in", torch.tensor([11, 22]))
        ds_f = LEGODataset(**lds_conf).data.f
        in_pdgid = ds_f.in_p[..., 0, -1].contiguous()
        out_pdgids = ds_f.out_p[..., -1:]

        self.counts = (out_pdgids == ptypes.view(1, 1, -1)).sum(1).clamp_max(max_particles - 1)
        self.pdgid_in_idx = torch.searchsorted(ptypes_in, in_pdgid)
        self.input = ds_f.in_cc.squeeze(-2)

        if use_density:
            self.input = torch.cat((ds_f.d.unsqueeze(-1), self.input), dim=-1)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return (
            self.input[idx].to(self.device),
            self.counts[idx].to(self.device),
            self.pdgid_in_idx[idx].to(self.device),
        )


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

        # Fused teacher-forcing embedding table: one Embedding of shape
        # ((max_seq_len - 1) * max_particles, h_dim). Per-position lookups
        # become a single op via the cached offset buffer.
        self.embd_in_ = torch.nn.Embedding(
            (rc.max_seq_len - 1) * rc.max_particles, rc.h_dim,
        )
        self.register_buffer(
            "_in_offsets",
            torch.arange(rc.max_seq_len - 1, dtype=torch.long) * rc.max_particles,
            persistent=False,
        )

        self.embd_pp_ = torch.nn.Embedding(rc.n_ptypes_in, rc.h_dim)

        # Fused per-position output projection: weight (L, H, P), bias (L, P).
        # Replaces ModuleList[Linear] + torch.stack with one einsum.
        self.proj_out_w = torch.nn.Parameter(
            torch.empty(rc.max_seq_len, rc.h_dim, rc.max_particles)
        )
        self.proj_out_b = torch.nn.Parameter(
            torch.empty(rc.max_seq_len, rc.max_particles)
        )
        # Mirror nn.Linear defaults per position (kaiming_uniform with
        # a=sqrt(5) on weight, uniform[-1/sqrt(fan_in), 1/sqrt(fan_in)] on bias).
        _bound = 1.0 / (rc.h_dim ** 0.5)
        for _i in range(rc.max_seq_len):
            torch.nn.init.kaiming_uniform_(self.proj_out_w[_i], a=5 ** 0.5)
            torch.nn.init.uniform_(self.proj_out_b[_i], -_bound, _bound)

        if rc.state_dict is not None:
            self.load_state_dict(rc.state_dict, strict=False)

        if rc.opt_conf is None:
            # Back-compat fallback for configs that don't set opt_conf.
            self.opt = schedulefree.AdamWScheduleFree(
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
        x[..., -6:] = self.geom_trafos.to_cube(x[..., -6:])
        x[..., -3:] = self.rc.pos_scale * x[..., -3:]
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
        in_cc, counts, pdgid_in_idx = batch
        in_embd = self.proj_in(in_cc)
        pdgid_embd = in_embd + self.embd_pp_(pdgid_in_idx)
        # Fused: shift per-position by max_particles, look up once.
        gt_embds = self.embd_in_(counts[:, : self.rc.max_seq_len - 1] + self._in_offsets)
        in_seq = torch.cat((pdgid_embd.unsqueeze(1), gt_embds), dim=1)
        out = self.model(in_seq, mask=None, condition=pdgid_embd)
        # Fused batched matmul replaces L separate Linear calls + stack.
        logits = torch.einsum("bsh,shp->bsp", out, self.proj_out_w) + self.proj_out_b

        loss_model = F.cross_entropy(
            logits.reshape(-1, self.rc.max_particles), counts.reshape(-1)
        )

        self.log("train_loss", loss_model, prog_bar=True, sync_dist=True)

        return loss_model

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
        in_cc, _, pdgid_in_idx = batch
        in_embd = self.proj_in(in_cc)
        x = (in_embd + self.embd_pp_(pdgid_in_idx)).unsqueeze(1)
        condition = x[:, 0]
        counts = in_cc.new_empty(x.shape[0], self.rc.max_seq_len, dtype=torch.long)

        cache = None
        for i in range(self.rc.max_seq_len):
            out, cache = self.model(
                x,
                mask=None,
                condition=condition,
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
