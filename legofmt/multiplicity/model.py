import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import json

from lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder

import schedulefree

from legofmt.data.dataloaders import LEGODataset
from legofmt.geometry.vmf_sampling import VMF


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
    def __init__(self, config: dict):
        super().__init__()
        state_dict = config.get("state_dict", None)
        if "config" in config.keys():
            config = config.get("config")
        self.config = config
        self.mm_conf = config.get("mm_conf", {})
        self.pos_scale = self.mm_conf.get("pos_scale", 50.0)
        h_dim = self.mm_conf.get("h_dim", 512)
        dl_conf = config.get("dl_conf", {})
        lds_conf = dl_conf.get("lds_args", {})
        if state_dict is None:
            with open(lds_conf.get("data") + "/meta.json") as f:
                meta_dict = json.load(f)
                self.mm_conf.setdefault("max_out_particles", meta_dict["ntokens"] - 3)
                self.mm_conf.setdefault(
                    "ptypes", torch.tensor(meta_dict["particles"]).sort().values
                )
                self.mm_conf.setdefault(
                    "ptypes_in", torch.tensor(meta_dict["particles_in"]).sort().values
                )
            if "max_count" not in self.mm_conf:
                _tmp_loader = MultLoader(self.config)
                self.mm_conf["max_count"] = int(_tmp_loader.counts.max().item()) + 1
                del _tmp_loader
        self.mm_conf.setdefault("max_count", self.mm_conf.get("max_out_particles"))
        self.max_particles = self.mm_conf["max_count"]
        dropout = self.mm_conf.get("dropout", 0.1)
        self.max_seq_len = self.mm_conf["ptypes"].shape[0]
        in_dim = self.mm_conf.get("in_dim", 6)
        self.n_ptypes_in = self.mm_conf.get("ptypes_in").shape[0]
        self.vmf = VMF()

        self.model = ContinuousTransformerWrapper(
            max_seq_len=self.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=self.mm_conf.get("use_abs_pos_emb", True),
            post_emb_norm=self.mm_conf.get("post_emb_norm", True),
            attn_layers=Decoder(
                dim=h_dim,
                depth=self.mm_conf.get("n_layers", 6),
                heads=self.mm_conf.get("n_heads", 8),
                layer_dropout=dropout,
                attn_dropout=dropout,
                ff_dropout=dropout,
                dim_condition=h_dim,
                **self.mm_conf.get("model_args", {}),
            ),
        )

        self.proj_in_ = torch.nn.Linear(in_dim, h_dim)

        self.embd_in_ = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.max_particles, h_dim)
                for _ in range(self.max_seq_len)
            ]
        )

        self.embd_pp_ = torch.nn.Embedding(self.n_ptypes_in, h_dim)

        self.proj_out_ = torch.nn.ModuleList(
            [
                torch.nn.Linear(h_dim, self.max_particles)
                for _ in range(self.max_seq_len)
            ]
        )

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

        opt_conf = self.mm_conf.get("opt_conf")
        if opt_conf is None:
            self.opt = schedulefree.AdamWScheduleFree(
                self.parameters(),
                lr=self.mm_conf.get("lr", 1e-3),
                betas=(0.95, 0.999),
                weight_decay=self.mm_conf.get("weight_decay", 0.0),
                warmup_steps=self.mm_conf.get("warmup_steps", 0),
            )
            self._sched_conf = None
        else:
            cfg = dict(opt_conf)
            opt_cls = cfg.pop("opt")
            self._sched_conf = cfg.pop("scheduler", None)
            self.opt = opt_cls(self.parameters(), **cfg)
        self._opt_is_sf = hasattr(self.opt, "train") and callable(getattr(self.opt, "train", None))

    def proj_in(self, x):
        x = x.clone()
        x[..., -6:] = self.vmf.to_cube(x[..., -6:])
        x[..., -3:] = self.pos_scale * x[..., -3:]
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
        gt_embds = torch.stack(
            [self.embd_in_[i](counts[:, i]) for i in range(self.max_seq_len - 1)], dim=1
        )
        in_seq = torch.cat((pdgid_embd.unsqueeze(1), gt_embds), dim=1)
        out = self.model(in_seq, mask=None, condition=pdgid_embd)
        logits = torch.stack(
            [self.proj_out_[i](out[:, i]) for i in range(self.max_seq_len)], dim=1
        )

        loss_model = F.cross_entropy(
            logits.reshape(-1, self.max_particles), counts.reshape(-1)
        )

        self.log("train_loss", loss_model, prog_bar=True, sync_dist=True)

        return loss_model

    def configure_optimizers(self):
        if self._sched_conf is None:
            return self.opt
        cfg = dict(self._sched_conf)
        sched_cls = cfg.pop("cls")
        interval = cfg.pop("interval", "step")
        scheduler = sched_cls(self.opt, **cfg)
        return {
            "optimizer": self.opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }

    def train_dataloader(self):
        dataset_train = MultLoader(self.config)
        num_workers = self.config.get("dl_conf").get("num_workers", 4)
        return DataLoader(
            dataset_train,
            batch_size=self.mm_conf.get("bs", 2**12),
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
        counts = in_cc.new_empty(x.shape[0], self.max_seq_len, dtype=torch.long)

        cache = None
        for i in range(self.max_seq_len):
            out, cache = self.model(
                x,
                mask=None,
                condition=condition,
                return_intermediates=True,
                cache=cache,
                input_not_include_cache=(i > 0),
            )
            logits = self.proj_out_[i](out[:, -1])
            sampled = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            x = self.embd_in_[i](sampled).unsqueeze(1)
            counts[:, i] = sampled

        return counts
