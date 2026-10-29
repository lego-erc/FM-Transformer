"""``LEGOLtng``: the Riemannian flow-matching LightningModule.

Holds construction, the Lightning hooks, and the helpers shared across its
mixins. The behaviour lives alongside: ``TrainStep`` (training step and loss),
``BaseDist`` (base distribution and its learned head), and ``Solvers`` (ODE
solving, likelihood, and ``forward``).
"""

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, random_split

import lightning as ltng

from legofmt.base_dist import base_nn
from legofmt.base_dist.base_nn import BaseDist, build_base_head, _OT_COUPLING_REQUIRES_LAP
from legofmt.base_dist.gen_base import GenerateBase

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.cfm.edep_ae import build_edep_ae
from legofmt.cfm.project_model import ProjectModel, uncompiled
from legofmt.cfm.solvers import Solvers

from legofmt.main.train_step import TrainStep

from legofmt.data.dataloaders import LEGODataset, LengthBucketBatchSampler, loader_seed, make_loader
from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F

from legofmt.cfm.path_sampler import ProductPathSampler
from legofmt.geometry.geom_trafos import zoom_polar
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer, opt_eval, opt_is_schedulefree, opt_train

from legofmt.log_metrics.val_metrics import ShowerValMetrics


class _Interacting:
    """Index view over the events that deposited energy, without copying them."""

    def __init__(self, ds, keep_idx: Tensor) -> None:
        self.ds, self.keep_idx = ds, keep_idx

    def __len__(self) -> int:
        return len(self.keep_idx)

    def __getitem__(self, i):
        return self.ds[self.keep_idx[i]]


def _event_lengths(ds) -> Tensor:
    if isinstance(ds, Subset):
        return _event_lengths(ds.dataset)[torch.as_tensor(ds.indices)]
    if isinstance(ds, _Interacting):
        return _event_lengths(ds.ds)[ds.keep_idx]
    return ds.data.am.full.sum(1)


def _interacting_only(ds) -> _Interacting:
    keep_idx = ((_F(ds.data.f.full).edep.reshape(-1) > 0)
                | (ds.data.am.out_p.sum(-1) != 1)).nonzero(as_tuple=True)[0]
    return _Interacting(ds, keep_idx)


class LEGOLtng(TrainStep, BaseDist, Solvers, ltng.LightningModule):

    def __init__(self, full_config: dict) -> None:
        super().__init__()
        self.rc = resolve_legoltng_config(full_config)

        types_embd = torch.arange(self.rc.max_seq_l, dtype=torch.int64).clamp_max(self.rc.n_prefix + 1).view(1, -1)

        self.register_buffer("pdgids_template", self.rc.pdgids_template)
        self.register_buffer("types_embd", types_embd)
        cuboid = torch.tensor(self.rc.config["model_conf"].get("cuboid_dim") or [1.0, 1.0, 1.0])
        self.register_buffer("cuboid_unit", cuboid / cuboid.max(), persistent=False)

        self.model       = self._build_model(self.rc)
        self.gen_base    = GenerateBase(self.rc.config)
        self.sym         = CubeSymmetry() if self.rc.canon_sym else None
        self.val_metrics = ShowerValMetrics()
        self.ps          = ProductPathSampler(self.rc.manifold)

        self._base_head_loss = None
        # frozen per-layer E_dep autoencoder: its K latents are flown as one token of the field (model_args.lat_dim);
        # solve leaves the generated latents in _edep_latents
        self.edep_ae = build_edep_ae(self.rc.edep_ae) if self.rc.edep_ae else None
        self._edep_dp = DataPrep(self.rc.config) if self.rc.edep_ae else None  # the layers' E_dep scale -> MeV
        self._edep_latents = None
        params = list(self.model.parameters())
        head = build_base_head(self.rc, self.gen_base)
        if head is not None:
            self.base_head = head
            params += [p for p in self.base_head.parameters() if p.requires_grad]
        elif self.rc.edep_whiten:
            raise ValueError("model_conf.edep_whiten needs the base head's per-event E_dep prior (base_pretrain_"
                             "batches > 0 or a saved base_conf.base_head, scale_dist sm_norm, Z/A/Size conditioning).")

        if self.rc.learned_loss_weights:
            self.lv = nn.Parameter(torch.zeros(3 + int(self.rc.edep_cell) + int(self.rc.n_lat > 0)))  # one per channel
            params += [self.lv]
            floor = torch.full_like(self.lv, self.rc.max_loss_weight).detach()
            if self.rc.edep_cell:
                floor[3] = self.rc.max_loss_weight_edep
            self.register_buffer("lv_floor", floor, persistent=False)
            if self.rc.one_step_euler_fac > 0:
                self.lv_flow = nn.Parameter(torch.zeros(1))
                params += [self.lv_flow]

        self._opt_params = params
        self.opt, self._lr_sched = build_optimizer(params, self.rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)  # read by lego_eval

        if self.rc.state_dict is not None:
            if self.rc.n_lat and self.rc.state_dict and "lat_in.weight" not in self.rc.state_dict:  # strict=False
                raise ValueError("model_conf.edep_ae checkpoint without the latent token (lat_in): the 16-row "
                                 "prototype (feat/edep-latents) is not supported.")
            self.model.vf.load_state_dict(self.rc.state_dict, strict=False)

        teacher = None
        if self.rc.reflow_path is not None:
            from legofmt.distill.reflow import _build_reflow_teacher  # avoids import cycle
            teacher = _build_reflow_teacher(self.rc.reflow_path)
        object.__setattr__(self, "reflow_teacher", teacher)  # not a submodule: no state_dict/DDP

    def _build_model(self, rc) -> nn.Module:
        return ProjectModel(CFMTrafo_x(**rc.model_args), rc.manifold)

    def _opt_train(self) -> None:
        opt_train(self.opt)

    def _opt_eval(self) -> None:
        opt_eval(self.opt)

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        self._opt_eval()

    def on_validation_model_train(self) -> None:
        super().on_validation_model_train()
        self._opt_train()

    @torch.no_grad()
    def on_fit_start(self) -> None:
        if self.rc.ot_coupling and base_nn.batch_linear_assignment is None:
            raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
        if self.reflow_teacher is not None:
            self.reflow_teacher.to(self.device)
        self.model.train()
        self._opt_train()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        start = self.rc.reflow_start_epoch
        if start <= 0 or self.current_epoch + 1 < start:
            return
        vf = uncompiled(self.model).vf
        if self.reflow_teacher is None:  # self-reflow: teacher = snapshot of the student
            from legofmt.distill.reflow import _teacher_from_state  # avoids import cycle
            teacher = _teacher_from_state(self.rc.config, vf.state_dict())
            object.__setattr__(self, "reflow_teacher", teacher.to(self.device))
        else:  # refresh: teacher = last epoch's student
            uncompiled(self.reflow_teacher.model).vf.load_state_dict(vf.state_dict())

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        """Raw pdgids -> 1-based indices into ``pdgids_template``; NaN, 0, ions
        (>= 1e8) and ids outside the vocabulary map to the unknown/pad index 0."""
        template  = self.pdgids_template.to(pdgids.device)
        pos       = torch.searchsorted(template, pdgids.contiguous()).clamp_max(len(template) - 1)
        unknown   = torch.isnan(pdgids) | (pdgids == 0) | (pdgids >= 1e8) | (template[pos] != pdgids)
        return (pos + 1).masked_fill_(unknown, 0)

    def _canon_dirs(self, x: Tensor, face: Tensor, fwd: Tensor, inverse: bool = False,
                    g: Tensor | None = None) -> Tensor:
        rot  = self.sym.uncanonicalize if inverse else self.sym.canonicalize
        dirs = rot(torch.stack((x[..., 1:4], x[..., 4:7]), dim=-2), face, g)
        new  = torch.cat((x[..., 0:1], dirs[..., 0, :], dirs[..., 1, :], x[..., 7:]), dim=-1)
        rows = torch.arange(x.shape[-2], device=x.device) >= self.rc.n_prefix
        new  = torch.where(rows.view(1, -1, 1), new, x)
        return torch.where(fwd[:, None, None], new, x)

    def _zoom_dirs(self, x: Tensor, am: Tensor, fwd: Tensor, inverse: bool = False,
                   pdgids: Tensor | None = None) -> Tensor:
        u    = x[..., self.rc.n_prefix:self.rc.n_prefix + 1, 1:4]
        dirs = zoom_polar(x[..., 1:4], u, self.rc.dir_zoom, inverse, self.rc.dir_zoom_mix)
        new  = torch.cat((x[..., 0:1], dirs, x[..., 4:]), dim=-1)
        rows = torch.arange(x.shape[-2], device=x.device) > self.rc.n_prefix
        if self.rc.dir_zoom_nucleons_max is not None:
            nucleons = torch.tensor([2112.0, 2212.0], device=x.device)
            if self.rc.pdgid_is_idx:
                nucleons = self.convert_pdgids(nucleons).to(pdgids.dtype)
            n_nuc = (am & rows & torch.isin(pdgids[..., 0], nucleons)).sum(-1)
            fwd = fwd & (n_nuc <= self.rc.dir_zoom_nucleons_max)
        return torch.where((am & rows & fwd[:, None]).unsqueeze(-1), new, x)

    def configure_optimizers(self):
        if self._lr_sched is not None:  # the trainer's exact count, not scripts/train.py's dataset_size estimate
            self.rc.opt_conf["scheduler"]["total_steps"] = int(self.trainer.estimated_stepping_batches)
            self.opt, self._lr_sched = build_optimizer(self._opt_params, self.rc.opt_conf)
        if self._lr_sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._lr_sched}

    def setup(self, stage: str | None = None) -> None:
        if getattr(self, "_val_ds", None) is not None:
            return
        full  = LEGODataset(**self.rc.dl_conf["lds_args"], prep=DataPrep(self.rc.config))
        if self.rc.config.get("model_conf", {}).get("exclude_passthrough", False):
            full = _interacting_only(full)
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen   = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(full, [len(full) - n_val, n_val], generator=gen)
        self._pretrain_base_if_needed()

    def _make_loader(self, dataset, *, shuffle: bool, bs: int | None = None, batch_sampler=None) -> DataLoader:
        return make_loader(dataset, bs=bs if bs is not None else self.rc.dl_conf.get("bs", 2**12), shuffle=shuffle,
                           num_workers=self.rc.dl_conf.get("num_workers", 4), batched_sampler=True,
                           batch_sampler=batch_sampler, seed=loader_seed(self.rc.dl_conf))

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.rc.len_buckets is not None:
            self._cpu_lengths = batch.am.full.sum(1)
            n = int(self._cpu_lengths.max())
            batch = batch.trim(next(b for b in self.rc.len_buckets if b >= n))
            if self.device.type == "cuda":
                batch = batch.pin_memory()
        return batch

    def train_dataloader(self) -> DataLoader:
        if self.rc.len_buckets is not None and self.rc.len_mode == "group":
            t = self._trainer
            return self._make_loader(self._train_ds, shuffle=True, batch_sampler=LengthBucketBatchSampler(
                _event_lengths(self._train_ds), self.rc.len_buckets, self.rc.dl_conf.get("bs", 2**12),
                seed=loader_seed(self.rc.dl_conf),
                rank=t.global_rank if t is not None else 0, world_size=t.world_size if t is not None else 1,
                max_tokens=self.rc.len_max_tokens,
            ))
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)
