"""``LEGOLtng``: the Riemannian flow-matching LightningModule.

The model is one class put together from mixins, one file each:

```
TrainStep                      main/train_step.py    the loss and the training step
BaseDist                       base/interface.py     the base (the distribution of x_0) and the base head
Solvers                        cfm/solvers.py        ODE solving, the likelihood, forward
Legacy*, ScheduleFreeSwitch    legacy/               superseded paths, each reached by one dispatch line
```

This file builds the parts, and holds the Lightning hooks and the token maps the mixins share:

```
convert_pdgids                 raw pdgid -> vocabulary index
species_idx                    the same, for tokens that may already hold indices
_enter_frame, _leave_frame     into and out of the frame the flow works in: the canonical cube frame
                               (_canon_dirs) and the zoomed polar angle (_zoom_dirs)
```

The config is checked where it is resolved (``legofmt/config.py``).
"""

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import lightning as ltng

from legofmt.base.ot_coupling import require_lap
from legofmt.base.interface import BaseDist, build_base_head
from legofmt.base.gen_base import GenerateBase
from legofmt.legacy.base_nn import LegacyBaseDist
from legofmt.legacy.edep_token import LegacyEdepToken
from legofmt.legacy.optimizers import ScheduleFreeSwitch, opt_is_schedulefree
from legofmt.legacy.train_step import LegacyTrainStep

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.cfm.edep_ae import build_edep_ae
from legofmt.cfm.project_model import ProjectModel
from legofmt.cfm.solvers import Solvers

from legofmt.main.train_step import TrainStep

from legofmt.data.load.packed import interacting_only, load_dataset
from legofmt.data.load.loader import (
    AUTO_MIN_BATCHES, LengthBucketBatchSampler, event_lengths, loader_seed, make_loader, smallest_bucket, split_val,
)
from legofmt.data.struct import DataStruct, _F, vocab_index, with_dirs
from legofmt.data.write.norm import DataPrep

from legofmt.cfm.path_sampler import ProductPathSampler
from legofmt.geometry.geom_trafos import zoom_polar
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.config import resolve_legoltng_config
from legofmt.optimizers import build_optimizer

from legofmt.main.val_metrics import ShowerValMetrics
from legofmt.rng import SYM_FLOW, randint_like


class LEGOLtng(TrainStep, LegacyTrainStep, LegacyEdepToken, LegacyBaseDist, BaseDist, Solvers, ScheduleFreeSwitch,
               ltng.LightningModule):
    """The flow: the field, its base, the training step and the solvers.

    Args:
        full_config: ``{"config": ..., "state_dict": ...}`` as a checkpoint stores it, or the config alone.
    """

    def __init__(self, full_config: dict) -> None:
        super().__init__()
        self.rc = rc = resolve_legoltng_config(full_config)

        # token types: one per prefix token, one for the incoming token, one shared by every outgoing slot
        types_embd = torch.arange(rc.max_seq_l, dtype=torch.int64).clamp(max=rc.n_prefix + 1).unsqueeze(0)
        self.register_buffer("pdgids_template", rc.pdgids_template)
        self.register_buffer("types_embd", types_embd)
        self.register_buffer("cuboid_unit", CubeTrace(rc.config["model_conf"].get("cuboid_dim")).cuboid_dim,
                             persistent=False)

        # the parts; reordering them changes the initial weights a seed gives
        self.model       = self._build_model(rc)
        self.gen_base    = GenerateBase(rc.config)
        self.sym         = CubeSymmetry() if rc.canon_sym else None
        self.val_metrics = ShowerValMetrics()
        self.ps          = ProductPathSampler(rc.manifold)
        # the frozen per-layer E_dep autoencoder: its K latents are one token of the field (model_args.lat_dim);
        # solve leaves the generated latents in _edep_latents
        self.edep_ae       = build_edep_ae(rc.edep_ae) if rc.edep_ae else None
        self._edep_dp      = DataPrep(rc.config) if rc.edep_ae else None   # E_dep and its layers: scales -> MeV
        self._edep_latents = None
        head = build_base_head(rc, self.gen_base)
        if head is not None:
            self.base_head = head

        # learned loss weights: one log-variance per loss cell (train_step.loss_cells), one for the flow-map term
        if rc.learned_loss_weights:
            self.lv = nn.Parameter(torch.zeros(3 + int(rc.edep_cell) + int(rc.n_lat > 0)))
            self.register_buffer("lv_floor", self._lv_floor(rc), persistent=False)
            if rc.one_step_euler_fac > 0:
                self.lv_flow = nn.Parameter(torch.zeros(1))

        self._opt_params = self._trained_params(head)
        self.opt, self._lr_sched = build_optimizer(self._opt_params, rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)  # read by lego_eval

        if rc.state_dict is not None:
            self.model.vf.load_state_dict(rc.state_dict, strict=False)
        self._init_reflow()   # legacy: the reflow teacher

    def _build_model(self, rc) -> nn.Module:
        """Constructs the field between its two projections (``ProjectModel``); ``LEGOLtngDirect`` overrides this."""
        return ProjectModel(CFMTrafo_x(**rc.model_args), rc.manifold, energy_border=rc.energy_border)

    # ---- token maps the mixins share ------------------------------------------------------------------------------

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        """Maps raw pdgids to vocabulary indices: 1-based positions in ``pdgids_template``, 0 for unknown or padding.

        NaN, 0, ions (>= 1e8) and every pdgid that is not in ``pdgids_template`` map to 0.
        """
        pos, known = vocab_index(self.pdgids_template.to(pdgids.device), pdgids.contiguous())
        unknown = (pdgids == 0) | (pdgids >= 1e8) | ~known   # NaN is never known
        return torch.where(unknown, 0, pos + 1)

    def species_idx(self, pdgids: Tensor) -> Tensor:
        """Returns the tokens' vocabulary indices.

        ``convert_pdgids`` of raw pdgids, or ``pdgids`` itself where the tokens already hold indices
        (``rc.pdgid_is_idx``, which ``GenerateOut`` sets for its batches).
        """
        return pdgids.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)

    def _enter_frame(self, ds_t: DataStruct, zoom: bool = False, aug: bool = False, rng=None,
                     stream: int = SYM_FLOW) -> tuple[DataStruct, tuple]:
        """Returns the batch in the frame the flow works in, and the frame, for ``_leave_frame``.

        ```
        canon_sym   every forward event rotated onto the canonical cube face; aug: a random symmetry g of that face
                    on top
        dir_zoom    the outgoing polar angles zoomed about the incoming direction (only with zoom)
        ```

        Args:
            ds_t:   the batch.
            zoom:   zoom the outgoing directions too. Training zooms its targets; generation does not, because the
                flow generates them zoomed.
            aug:    sample ``g`` (``sym_aug``).
            rng:    keys ``g`` on the event (``legofmt.rng.EventRNG``), or None.
            stream: the draw site's id (``legofmt.rng``), which a keyed ``g`` uses.

        Returns:
            ``(ds_t, (face, g, fwd))``: the batch in the frame (without its ``edep_layers`` once it is rotated or
            zoomed); (B,), each event's entry face (None without ``canon_sym``); (B,), each event's ``g`` in 0..7
            (None without ``aug``); (B,), the forward events.
        """
        fwd = ds_t.m.fwd
        face = g = None
        if self.sym is not None:
            face = self.sym.face_of(ds_t.f.in_pos)
            g    = randint_like(8, face, rng, stream) if aug else None
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd, g=g), ds_t.m.full, ds_t.am.full)
        if zoom and self.rc.dir_zoom:
            ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, ds_t.am.full, ds_t.m.fwd, pdgids=ds_t.f.pdgids),
                              ds_t.m.full, ds_t.am.full)
        return ds_t, (face, g, fwd)

    def _leave_frame(self, x: Tensor, ds_t: DataStruct, frame: tuple) -> Tensor:
        """Returns the tokens ``x`` (B, L, 7 or 8) out of the flow's frame: ``dir_zoom`` undone, then ``canon_sym``.

        ``ds_t`` and ``frame`` as ``_enter_frame`` returned them.
        """
        face, g, fwd = frame
        if self.rc.dir_zoom:
            x = self._zoom_dirs(x, ds_t.am.full, fwd, inverse=True, pdgids=ds_t.f.pdgids)
        return x if face is None else self._canon_dirs(x, face, fwd, inverse=True, g=g)

    def _canon_dirs(self, x: Tensor, face: Tensor, fwd: Tensor, inverse: bool = False,
                    g: Tensor | None = None) -> Tensor:
        """Rotates the momentum and position directions onto the canonical cube face (``canon_sym``), or back.

        Only the incoming and outgoing tokens of forward events are rotated.

        Args:
            x:       (B, L, 7 or 8), the tokens, with or without the pdgid channel.
            face:    (B,), the face each event enters through (``CubeSymmetry.face_of``).
            fwd:     (B,), the forward events.
            inverse: rotate back from the canonical frame.
            g:       (B,), each event's symmetry of the canonical face, in 0..7 (``sym_aug``), or None.

        Returns:
            The rotated tokens.
        """
        rot = self.sym.uncanonicalize if inverse else self.sym.canonicalize
        mom, pos = rot(torch.stack((_F(x).mom, _F(x).pos), dim=-2), face, g).unbind(-2)
        particles = torch.arange(x.shape[-2], device=x.device) >= self.rc.n_prefix
        return with_dirs(x, particles & fwd.unsqueeze(1), mom, pos)

    def _zoom_dirs(self, x: Tensor, am: Tensor, fwd: Tensor, inverse: bool = False,
                   pdgids: Tensor | None = None) -> Tensor:
        """Zooms each outgoing momentum's polar angle about the incoming direction (``dir_zoom``), or back.

        Only the real outgoing tokens of forward events change; with ``dir_zoom_nucleons_max`` only in events with
        at most that many outgoing nucleons.

        Args:
            x:       (B, L, 7 or 8), the tokens, with or without the pdgid channel.
            am:      (B, L), the real tokens (the attention mask).
            fwd:     (B,), the forward events.
            inverse: zoom back.
            pdgids:  (B, L, 1), the tokens' species, for the nucleon count (``dir_zoom_nucleons_max``).

        Returns:
            The zoomed tokens.
        """
        zoomed   = zoom_polar(_F(x).mom, _F(x).in_p[..., 1:4], self.rc.dir_zoom, inverse, self.rc.dir_zoom_mix)
        outgoing = am & (torch.arange(x.shape[-2], device=x.device) > self.rc.n_prefix)
        if self.rc.dir_zoom_nucleons_max is not None:
            fwd = fwd & (self._n_nucleons(pdgids, outgoing) <= self.rc.dir_zoom_nucleons_max)
        return with_dirs(x, outgoing & fwd.unsqueeze(1), mom=zoomed)

    # ---- Lightning: data --------------------------------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Loads the dataset once, splits off the validation events, and pretrains the base head if it is not frozen."""
        if getattr(self, "_val_ds", None) is not None:
            return
        full = load_dataset(**self.rc.dl_conf["lds_args"], prep=DataPrep(self.rc.config))
        pool = self.model.vf.cond_pool
        # a fresh field with ``cond_pool`` standardises its conditioning scalars by this dataset's mean and std
        if pool is not None and self.rc.state_dict is None:
            pool.standardise_by(torch.stack([full.cond(name) for name in self.rc.cond_scalars], dim=-1))
        if self.rc.config.get("model_conf", {}).get("exclude_passthrough", False):
            full = interacting_only(full)
        self._train_ds, self._val_ds = split_val(full, self.rc.val_conf)
        self._pretrain_base_if_needed()

    def train_dataloader(self) -> DataLoader:
        """Returns the training loader; ``len_mode: group`` batches events of one length bucket together."""
        if self.rc.len_buckets is None or self.rc.len_mode != "group":
            return self._make_loader(self._train_ds, shuffle=True)
        return self._make_loader(self._train_ds, shuffle=True, batch_sampler=self._bucket_sampler())

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        """Cuts the batch to its length bucket before it is moved to the device.

        The events' lengths are kept on the CPU (``_cpu_lengths``): ``_len_pieces`` reads them there, without a GPU
        sync.
        """
        if self.rc.len_buckets is not None:
            self._cpu_lengths = batch.am.full.sum(1)
            batch = batch.trim(int(smallest_bucket(self._cpu_lengths.max(), self.rc.len_buckets)))
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

    @torch.no_grad()
    def on_fit_start(self) -> None:
        if self.rc.ot_coupling: require_lap()
        if self.reflow_teacher is not None: self.reflow_teacher.to(self.device)   # legacy: reflow
        self.model.train()
        self._opt_train()   # legacy: schedule-free

    # ---- plumbing ---------------------------------------------------------------------------------------------------

    def _lv_floor(self, rc) -> Tensor:
        """Returns the floor of each loss cell's log-variance ``lv_c``, which caps its weight ``exp(-lv_c)``.

        ``max_loss_weight`` for every cell but E_dep's, which has ``max_loss_weight_edep``.
        """
        floor = torch.full_like(self.lv, rc.max_loss_weight).detach()
        if rc.edep_cell:
            floor[3] = rc.max_loss_weight_edep
        return floor

    def _trained_params(self, head: nn.Module | None) -> list:
        """Returns the parameters the optimizer trains, in order: the field's, the base head's, the loss weights."""
        params = list(self.model.parameters())
        if head is not None:
            params += [p for p in head.parameters() if p.requires_grad]
        if self.rc.learned_loss_weights:
            params += [self.lv]
            if self.rc.one_step_euler_fac > 0:
                params += [self.lv_flow]
        return params

    def _n_nucleons(self, pdgids: Tensor, tokens: Tensor) -> Tensor:
        """Returns each event's number of protons and neutrons among ``tokens`` (B, L); ``pdgids`` (B, L, 1)."""
        nucleons = torch.tensor([2112.0, 2212.0], device=pdgids.device)
        if self.rc.pdgid_is_idx:
            nucleons = self.convert_pdgids(nucleons).to(pdgids.dtype)
        return (tokens & torch.isin(pdgids[..., 0], nucleons)).sum(-1)

    def _bucket_sampler(self) -> LengthBucketBatchSampler:
        """Returns the sampler that batches the training events of one length bucket together, per DDP rank."""
        t = self._trainer
        return LengthBucketBatchSampler(
            event_lengths(self._train_ds), self.rc.len_buckets, self.rc.dl_conf.get("bs", 2**12),
            seed=loader_seed(self.rc.dl_conf),
            rank=t.global_rank if t is not None else 0, world_size=t.world_size if t is not None else 1,
            max_tokens=self.rc.len_max_tokens,
            min_batches=AUTO_MIN_BATCHES if self.rc.dl_conf.get("len_buckets") == "auto" else 0,
        )

    def _make_loader(self, dataset, *, shuffle: bool, bs: int | None = None, batch_sampler=None) -> DataLoader:
        """Constructs the DataLoader (``make_loader``) with ``dl_conf``'s batch size, workers and seed."""
        return make_loader(dataset, bs=bs if bs is not None else self.rc.dl_conf.get("bs", 2**12), shuffle=shuffle,
                           num_workers=self.rc.dl_conf.get("num_workers", 4), batch_sampler=batch_sampler,
                           seed=loader_seed(self.rc.dl_conf))
