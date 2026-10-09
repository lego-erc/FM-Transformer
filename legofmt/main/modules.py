"""``LEGOLtng``: the Riemannian flow-matching LightningModule.

Holds construction, the Lightning hooks, and the helpers shared across its
mixins. The behaviour lives alongside: ``TrainStep`` (training step and loss),
``BaseDist`` (base distribution and its learned head), and ``Solvers`` (ODE
solving, likelihood, and ``forward``).
"""

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

import lightning as ltng

from legofmt.base_dist import base_nn
from legofmt.base_dist.base_nn import (
    BaseDist, build_base_head, _OT_COUPLING_REQUIRES_LAP,
)
from legofmt.base_dist.gen_base import GenerateBase

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.cfm.project_model import ProjectModel, uncompiled
from legofmt.cfm.solvers import Solvers

from legofmt.main.train_step import TrainStep

from legofmt.data.dataloaders import LEGODataset, make_loader
from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F

from legofmt.geometry.geom_trafos import GeomTrafos

from legofmt.cfm.path_sampler import ProductPathSampler
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import (
    build_optimizer, opt_eval, opt_is_schedulefree, opt_train,
)

from legofmt.log_metrics.val_metrics import ShowerValMetrics


class _Interacting:
    """Index view over the events that deposited energy, without copying them."""

    def __init__(self, ds, keep_idx: Tensor) -> None:
        self.ds, self.keep_idx = ds, keep_idx

    def __len__(self) -> int:
        return len(self.keep_idx)

    def __getitem__(self, i):
        return self.ds[self.keep_idx[i]]


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

        self.model       = self._build_model(self.rc)
        self.gen_base    = GenerateBase(self.rc.config)
        self.geom_trafos = GeomTrafos()
        self.sym         = CubeSymmetry() if self.rc.canon_sym else None
        self.val_metrics = ShowerValMetrics()
        self.ps          = ProductPathSampler(self.rc.manifold)

        self.register_buffer(
            "_pt_allowed",
            torch.tensor([int(p) in set(self.rc.passthrough_pdgids)
                          for p in [0, *self.rc.pdgids_template.tolist()]]),
            persistent=False,
        )

        self._base_head_loss = None
        params = list(self.model.parameters())
        head = build_base_head(self.rc, self.gen_base)
        if head is not None:
            self.base_head = head
            params += [p for p in self.base_head.parameters() if p.requires_grad]

        if self.rc.learned_loss_weights:
            self.lv = nn.Parameter(torch.zeros(3 + int(self.rc.edep_cell)))  # one cell per channel
            params += [self.lv]
            if self.rc.one_step_euler_fac > 0:
                self.lv_flow = nn.Parameter(torch.zeros(1))
                params += [self.lv_flow]

        self.opt, self._lr_sched = build_optimizer(params, self.rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)  # read by lego_eval

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
        )

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
        vf = uncompiled(self.model).vf
        if self.reflow_teacher is None:  # self-reflow: teacher = snapshot of the student
            from legofmt.distill.reflow import _teacher_from_state  # avoids import cycle
            teacher = _teacher_from_state(self.rc.config, vf.state_dict())
            object.__setattr__(self, "reflow_teacher", teacher.to(self.device))
        else:  # refresh: teacher = last epoch's student
            t_m = self.reflow_teacher.model
            uncompiled(t_m).vf.load_state_dict(vf.state_dict())

    @property
    def pt_head(self) -> nn.Module:
        """The pass-through gate. It lives on ``vf`` so that it rides the only
        state_dict ``scripts/train.py`` saves for the flow."""
        return uncompiled(self.model).vf.pt_head

    def _in_species(self, ds_t) -> Tensor:
        """Incoming species as a vocabulary index, pad/unknown clamped to 0."""
        pid = ds_t.f.in_p[..., 0, -1]
        idx = pid.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pid)
        return idx.long().clamp(0, len(self.pdgids_template))

    def pt_logits(self, ds_t, idx: Tensor | None = None) -> Tensor:
        """Pass-through logit per event, from what MultModel's pass-through token saw:
        the conditioning scalars and the incoming particle (position half mapped to
        the cube), plus the incoming species."""
        f    = ds_t.f
        inc  = f.in_cc[..., 0, :].nan_to_num(1.0)
        inc  = torch.cat((inc[..., 0:1], self.geom_trafos.to_cube(inc[..., 1:7])), dim=-1)
        oh   = nn.functional.one_hot(
            self._in_species(ds_t) if idx is None else idx, len(self.pdgids_template) + 1)
        cond = torch.cat([f.cond(n).view(-1, 1) for n in self.rc.cond_scalars], dim=-1)
        x    = torch.cat((cond, inc, oh.to(inc.dtype)), dim=-1)
        return self.pt_head(x).squeeze(-1)

    def pt_allowed(self, ds_t, idx: Tensor | None = None) -> Tensor:
        """Events whose primary is a species that can pass through at all."""
        return self._pt_allowed[self._in_species(ds_t) if idx is None else idx]

    @torch.no_grad()
    def sample_passthrough(self, ds_t) -> Tensor:
        """Bernoulli draw of "this primary did not interact", masked to the species
        that physically can. Answered before the solve, so a fired event skips it."""
        idx = self._in_species(ds_t)
        p   = self.pt_logits(ds_t, idx).sigmoid()
        return (torch.rand_like(p) < p) & self.pt_allowed(ds_t, idx)

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

    def configure_optimizers(self):
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
        self._train_ds, self._val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=gen,
        )
        self._pretrain_base_if_needed()

    def _make_loader(self, dataset, *, shuffle: bool, bs: int | None = None) -> DataLoader:
        return make_loader(
            dataset,
            bs=bs if bs is not None else self.rc.dl_conf.get("bs", 2**12),
            shuffle=shuffle,
            num_workers=self.rc.dl_conf.get("num_workers", 4),
            batched_sampler=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)
