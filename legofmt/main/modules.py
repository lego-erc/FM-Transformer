"""``LEGOLtng``: the Riemannian flow-matching LightningModule.

The model is one class put together from mixins, one file each:

```
TrainStep    main/train_step.py    the loss and the training step
BaseDist     base/interface.py     the base distribution x_0 and its learned head
Solvers      cfm/solvers.py        ODE solving, the likelihood, forward
Legacy*      legofmt/legacy/       superseded paths, each reached by one dispatch line
```

This file builds the parts, holds the token maps the mixins share (``convert_pdgids``: raw pdgid -> vocabulary
index, ``_canon_dirs``: the canonical cube frame, ``_zoom_dirs``: the zoomed polar angle) and the Lightning hooks.
The config is checked where it is resolved (``mod_comps/config.py``).
"""

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

import lightning as ltng

from legofmt.base import ot_coupling
from legofmt.base.interface import BaseDist, build_base_head
from legofmt.base.gen_base import GenerateBase
from legofmt.legacy.base_nn import LegacyBaseDist
from legofmt.legacy.edep_token import LegacyEdepToken
from legofmt.legacy.train_step import LegacyTrainStep

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.cfm.edep_ae import build_edep_ae
from legofmt.cfm.project_model import ProjectModel
from legofmt.cfm.solvers import Solvers

from legofmt.main.train_step import TrainStep

from legofmt.data.load.packed import interacting_only, load_dataset
from legofmt.data.load.loader import LengthBucketBatchSampler, event_lengths, loader_seed, make_loader
from legofmt.data.struct import CHANNELS, _F
from legofmt.data.write.norm import DataPrep

from legofmt.cfm.path_sampler import ProductPathSampler
from legofmt.geometry.geom_trafos import zoom_polar
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer, opt_eval, opt_is_schedulefree, opt_train

from legofmt.log_metrics.val_metrics import ShowerValMetrics


class LEGOLtng(TrainStep, LegacyTrainStep, LegacyEdepToken, LegacyBaseDist, BaseDist, Solvers, ltng.LightningModule):
    """The flow-matching model: the field, its base distribution, training and solving.

    Args:
        full_config: ``{"config": ..., "state_dict": ...}`` as a checkpoint stores it, or the config alone.
    """

    def __init__(self, full_config: dict) -> None:
        super().__init__()
        self.rc = rc = resolve_legoltng_config(full_config)

        # token types: one per prefix token, the incoming one, then one shared by every outgoing slot
        types_embd = torch.arange(rc.max_seq_l, dtype=torch.int64).clamp_max(rc.n_prefix + 1).view(1, -1)
        self.register_buffer("pdgids_template", rc.pdgids_template)
        self.register_buffer("types_embd", types_embd)
        cuboid = torch.tensor(rc.config["model_conf"].get("cuboid_dim") or [1.0, 1.0, 1.0])
        self.register_buffer("cuboid_unit", cuboid / cuboid.max(), persistent=False)

        # the parts, in parameter-initialisation order
        self.model       = self._build_model(rc)
        self.gen_base    = GenerateBase(rc.config)
        self.sym         = CubeSymmetry() if rc.canon_sym else None
        self.val_metrics = ShowerValMetrics()
        self.ps          = ProductPathSampler(rc.manifold)
        # the frozen per-layer E_dep autoencoder: its K latents are one token of the field (model_args.lat_dim);
        # solve leaves the generated latents in _edep_latents
        self.edep_ae       = build_edep_ae(rc.edep_ae) if rc.edep_ae else None
        self._edep_dp      = DataPrep(rc.config) if rc.edep_ae else None   # the layers' E_dep scale -> MeV
        self._edep_latents = None
        params = list(self.model.parameters())
        head = build_base_head(rc, self.gen_base)
        if head is not None:
            self.base_head = head
            params += [p for p in head.parameters() if p.requires_grad]

        # learned loss weights: one log-variance per loss cell (train_step.loss_cells), one for the flow-map term
        if rc.learned_loss_weights:
            self.lv = nn.Parameter(torch.zeros(3 + int(rc.edep_cell) + int(rc.n_lat > 0)))
            params += [self.lv]
            floor = torch.full_like(self.lv, rc.max_loss_weight).detach()
            if rc.edep_cell:
                floor[3] = rc.max_loss_weight_edep
            self.register_buffer("lv_floor", floor, persistent=False)
            if rc.one_step_euler_fac > 0:
                self.lv_flow = nn.Parameter(torch.zeros(1))
                params += [self.lv_flow]

        self._opt_params = params
        self.opt, self._lr_sched = build_optimizer(params, rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)  # read by lego_eval

        if rc.state_dict is not None:
            self.model.vf.load_state_dict(rc.state_dict, strict=False)
        self._init_reflow()   # legacy: the reflow teacher

    def _build_model(self, rc) -> nn.Module:
        """Builds the field on the manifold (``LEGOLtngDirect`` swaps in its one-step model)."""
        return ProjectModel(CFMTrafo_x(**rc.model_args), rc.manifold)

    # ---- token maps the mixins share ------------------------------------------------------------------------------

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        """Maps raw pdgids to 1-based indices into ``pdgids_template``.

        NaN, 0, ions (>= 1e8) and ids outside the vocabulary map to the unknown/pad index 0.
        """
        template = self.pdgids_template.to(pdgids.device)
        pos      = torch.searchsorted(template, pdgids.contiguous()).clamp_max(len(template) - 1)
        unknown  = (pdgids == 0) | (pdgids >= 1e8) | (template[pos] != pdgids)   # NaN != every entry
        return (pos + 1).masked_fill_(unknown, 0)

    def _canon_dirs(self, x: Tensor, face: Tensor, fwd: Tensor, inverse: bool = False,
                    g: Tensor | None = None) -> Tensor:
        """Rotates the momentum and position directions onto the canonical cube face (``canon_sym``), or back.

        Only the particle tokens of forward events turn.

        Args:
            x:       (B, L, 8) tokens.
            face:    (B,), the face each event enters through (``CubeSymmetry.face_of``).
            fwd:     (B,), the forward events.
            inverse: rotate back from the canonical frame.
            g:       (B,), a cube symmetry on top (``sym_aug``), or None.

        Returns:
            The rotated tokens.
        """
        rot = self.sym.uncanonicalize if inverse else self.sym.canonicalize
        energy, mom, pos, pid = x.tensor_split(CHANNELS, dim=-1)
        new = torch.cat((energy, rot(torch.stack((mom, pos), dim=-2), face, g).flatten(-2), pid), dim=-1)
        particles = torch.arange(x.shape[-2], device=x.device) >= self.rc.n_prefix
        return torch.where(particles.view(1, -1, 1) & fwd[:, None, None], new, x)

    def _zoom_dirs(self, x: Tensor, am: Tensor, fwd: Tensor, inverse: bool = False,
                   pdgids: Tensor | None = None) -> Tensor:
        """Zooms each outgoing momentum's polar angle about the incoming direction (``dir_zoom``), or back.

        Only the real outgoing tokens of forward events change; with ``dir_zoom_nucleons_max`` only events with at
        most that many nucleons out.

        Args:
            x:       (B, L, 8) tokens.
            am:      (B, L), the real tokens.
            fwd:     (B,), the forward events.
            inverse: zoom back.
            pdgids:  (B, L, 1), the tokens' species, for the nucleon count.

        Returns:
            The zoomed tokens.
        """
        energy, mom, pos, pid = x.tensor_split(CHANNELS, dim=-1)
        zoomed   = zoom_polar(mom, _F(x).in_p[..., 1:4], self.rc.dir_zoom, inverse, self.rc.dir_zoom_mix)
        new      = torch.cat((energy, zoomed, pos, pid), dim=-1)
        outgoing = am & (torch.arange(x.shape[-2], device=x.device) > self.rc.n_prefix)
        if self.rc.dir_zoom_nucleons_max is not None:
            nucleons = torch.tensor([2112.0, 2212.0], device=x.device)
            if self.rc.pdgid_is_idx:
                nucleons = self.convert_pdgids(nucleons).to(pdgids.dtype)
            fwd = fwd & ((outgoing & torch.isin(pdgids[..., 0], nucleons)).sum(-1) <= self.rc.dir_zoom_nucleons_max)
        return torch.where((outgoing & fwd[:, None]).unsqueeze(-1), new, x)

    # ---- Lightning: data --------------------------------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Loads the dataset once, splits off the validation events, and fits the base head if it needs it."""
        if getattr(self, "_val_ds", None) is not None:
            return
        full = load_dataset(**self.rc.dl_conf["lds_args"], prep=DataPrep(self.rc.config))
        if self.rc.config.get("model_conf", {}).get("exclude_passthrough", False):
            full = interacting_only(full)
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen   = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(full, [len(full) - n_val, n_val], generator=gen)
        self._pretrain_base_if_needed()

    def train_dataloader(self) -> DataLoader:
        """Returns the training loader; ``len_mode: group`` batches events of one length bucket together."""
        if self.rc.len_buckets is None or self.rc.len_mode != "group":
            return self._make_loader(self._train_ds, shuffle=True)
        t = self._trainer
        return self._make_loader(self._train_ds, shuffle=True, batch_sampler=LengthBucketBatchSampler(
            event_lengths(self._train_ds), self.rc.len_buckets, self.rc.dl_conf.get("bs", 2**12),
            seed=loader_seed(self.rc.dl_conf),
            rank=t.global_rank if t is not None else 0, world_size=t.world_size if t is not None else 1,
            max_tokens=self.rc.len_max_tokens,
        ))

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def _make_loader(self, dataset, *, shuffle: bool, bs: int | None = None, batch_sampler=None) -> DataLoader:
        """Returns ``make_loader`` with the ``dl_conf`` batch size, workers and seed."""
        return make_loader(dataset, bs=bs if bs is not None else self.rc.dl_conf.get("bs", 2**12), shuffle=shuffle,
                           num_workers=self.rc.dl_conf.get("num_workers", 4), batch_sampler=batch_sampler,
                           seed=loader_seed(self.rc.dl_conf))

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        """Cuts the batch to its length bucket before the upload; its lengths stay on the host for ``_len_pieces``."""
        if self.rc.len_buckets is not None:
            self._cpu_lengths = batch.am.full.sum(1)
            n = int(self._cpu_lengths.max())
            batch = batch.trim(next(b for b in self.rc.len_buckets if b >= n))
            if self.device.type == "cuda":
                batch = batch.pin_memory()
        return batch

    # ---- Lightning: optimizer ---------------------------------------------------------------------------------------

    def configure_optimizers(self):
        """Returns the optimizer, its schedule rebuilt on the trainer's exact step count."""
        if self._lr_sched is not None:  # the trainer's exact count, not scripts/train.py's dataset_size estimate
            self.rc.opt_conf["scheduler"]["total_steps"] = int(self.trainer.estimated_stepping_batches)
            self.opt, self._lr_sched = build_optimizer(self._opt_params, self.rc.opt_conf)
        if self._lr_sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._lr_sched}

    def _opt_train(self) -> None:
        opt_train(self.opt)

    def _opt_eval(self) -> None:
        opt_eval(self.opt)

    @torch.no_grad()
    def on_fit_start(self) -> None:
        if self.rc.ot_coupling and ot_coupling.batch_linear_assignment is None:
            raise RuntimeError(ot_coupling._OT_COUPLING_REQUIRES_LAP)
        if self.reflow_teacher is not None: self.reflow_teacher.to(self.device)   # legacy: reflow
        self.model.train()
        self._opt_train()

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        self._opt_eval()

    def on_validation_model_train(self) -> None:
        super().on_validation_model_train()
        self._opt_train()
