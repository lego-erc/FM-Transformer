import torch
from torch import Tensor, nn
from torch.utils.data import (
    BatchSampler, DataLoader, RandomSampler, SequentialSampler, random_split,
)

import lightning as ltng

from legofmt.base_dist import base_nn
from legofmt.base_dist.base_nn import (
    BaseDist, build_base_head, _OT_COUPLING_REQUIRES_LAP,
)
from legofmt.base_dist.gen_base import GenerateBase

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.cfm.project_model import ProjectModel
from legofmt.cfm.solvers import Solvers

from legofmt.main.train_step import TrainStep

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.prep import DataPrep
from legofmt.data.struct import DataStruct

from legofmt.geometry.path_sample_mult import ProductPathSampler
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer

from legofmt.log_metrics.val_metrics import ShowerValMetrics


class LEGOLtng(TrainStep, BaseDist, Solvers, ltng.LightningModule):

    def __init__(self, full_config: dict) -> None:
        super().__init__()
        self.rc = resolve_legoltng_config(full_config)

        types_embd = torch.arange(self.rc.max_seq_l, dtype=torch.int64).clamp_max(self.rc.n_prefix + 1).view(1, -1)

        self.register_buffer("pdgids_template", self.rc.pdgids_template)
        self.register_buffer("types_embd", types_embd)

        self.model       = self._build_model(self.rc)
        self.gen_base    = GenerateBase(self.rc.config)
        self.sym         = CubeSymmetry() if self.rc.canon_sym else None
        self.val_metrics = ShowerValMetrics()
        self.loss_fn     = nn.MSELoss()
        self.ps          = ProductPathSampler(self.rc.manifold)

        self._base_dist_loss = None
        params = list(self.model.parameters())
        head = build_base_head(self.rc, self.gen_base)
        if head is not None:
            self.base_head = head
            params += [p for p in self.base_head.parameters() if p.requires_grad]
        if self.rc.uncert_weighting:
            self.lv = nn.Parameter(torch.zeros(self.rc.uncert_bins * 3))
            params += [self.lv]
            if self.rc.one_step_euler_fac > 0:
                self.lv_flow = nn.Parameter(torch.zeros(1))
                params += [self.lv_flow]

        self.opt, self._lr_sched = build_optimizer(params, self.rc.opt_conf)
        self._opt_is_sf = callable(getattr(self.opt, "train", None))

        if self.rc.state_dict is not None:
            self.model.vf.load_state_dict(self.rc.state_dict, strict=False)

        teacher = None
        if self.rc.reflow_path is not None:
            from legofmt.distill.reflow import _build_reflow_teacher  # avoids import cycle
            teacher = _build_reflow_teacher(self.rc.reflow_path)
        object.__setattr__(self, "reflow_teacher", teacher)  # not a submodule: no state_dict/DDP

    def _build_model(self, rc) -> nn.Module:
        return ProjectModel(
            CFMTrafo_x(**rc.model_args),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )

    def _opt_train(self) -> None:
        if getattr(self, "_opt_is_sf", False):
            self.opt.train()

    def _opt_eval(self) -> None:
        if getattr(self, "_opt_is_sf", False):
            self.opt.eval()

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        self._opt_eval()

    def on_validation_model_train(self) -> None:
        super().on_validation_model_train()
        self._opt_train()

    @torch.no_grad()
    def on_fit_start(self) -> None:
        if self.rc.ot_coupling and base_nn.slap is None:
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
        vf = (self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model).vf
        if self.reflow_teacher is None:  # self-reflow: teacher = snapshot of the student
            from legofmt.distill.reflow import _teacher_from_state  # avoids import cycle
            teacher = _teacher_from_state(self.rc.config, vf.state_dict())
            object.__setattr__(self, "reflow_teacher", teacher.to(self.device))
        else:  # refresh: teacher = last epoch's student
            t_m = self.reflow_teacher.model
            (t_m._orig_mod if hasattr(t_m, "_orig_mod") else t_m).vf.load_state_dict(vf.state_dict())

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        cond = torch.isnan(pdgids) | (pdgids == 0) | (pdgids >= 1e8)
        pdgid_idx = torch.searchsorted(
            self.pdgids_template.to(pdgids.device), pdgids.contiguous()) + 1
        return pdgid_idx.masked_fill_(cond, 0)

    def _canon_dirs(self, x: Tensor, face: Tensor, fwd: Tensor, inverse: bool = False) -> Tensor:
        rot  = self.sym.uncanonicalize if inverse else self.sym.canonicalize
        dirs = rot(torch.stack((x[..., 1:4], x[..., 4:7]), dim=-2), face)
        new  = torch.cat((x[..., 0:1], dirs[..., 0, :], dirs[..., 1, :], x[..., 7:]), dim=-1)
        rows = torch.arange(x.shape[-2], device=x.device) >= self.rc.n_prefix
        new  = torch.where(rows.view(1, -1, 1), new, x)
        return torch.where(fwd[:, None, None], new, x)

    def configure_optimizers(self):
        if self._lr_sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._lr_sched}

    def setup(self, stage: str | None = None) -> None:
        if getattr(self, "_val_ds", None) is not None:
            return
        full  = LEGODataset(**self.rc.dl_conf["lds_args"], prep=DataPrep(self.rc.config))
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen   = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=gen,
        )
        self._pretrain_base_if_needed()

    def _make_loader(self, dataset, *, shuffle: bool, bs: int | None = None) -> DataLoader:
        num_workers = self.rc.dl_conf.get("num_workers", 4)
        sampler = (RandomSampler if shuffle else SequentialSampler)(dataset)
        return DataLoader(
            dataset,
            sampler=BatchSampler(
                sampler,
                bs if bs is not None else self.rc.dl_conf.get("bs", 2**12),
                drop_last=False,
            ),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)
