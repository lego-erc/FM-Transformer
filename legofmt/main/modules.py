import warnings
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

import lightning as ltng

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

try:
    from torch_lap_cuda_lib import solve_lap as slap
except ImportError:
    slap = None

_OT_COUPLING_REQUIRES_LAP = (
    "ot_coupling=True requires `torch_lap_cuda_lib`. "
    "Install it or set model_conf.ot_coupling=False."
)

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.struct import DataStruct, _F
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.geometry.gen_base import GenerateBase
from legofmt.geometry.path_sample_mult import ProductPathSampler, ProductManifold
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer
from legofmt.log_metrics.val_metrics import ShowerValMetrics


class ProjectModel(ModelWrapper, nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(
        self,
        vf: nn.Module,
        manifold: ProductManifold,
        **kwargs: dict,
    ) -> None:
        """Wraps a velocity field with manifold projection.

        Args:
            vf: the factorized transformer (:class:`CFMTrafo_x`).
            manifold: product manifold the features live on.
            **kwargs: reads ``cond_cube`` and ``no_detach``; the rest are
                retained on :attr:`kwargs`.
        """
        nn.Module.__init__(self)
        self.vf = vf
        self.manifold = manifold
        self.kwargs = kwargs
        self.geom_trafos = GeomTrafos()
        self.cond_cube = kwargs.get("cond_cube", False)
        self.no_detach = kwargs.get("no_detach", False)

    def _prep_x(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects ``x`` onto the manifold at attended slots.

        Returns:
            tuple: ``(x_proj_full, x_attended, x_surr)`` -- the fully
            projected features, the attend-gated (optionally detached)
            features, and the surrogate fed to the transformer
            (cube-projected position when :attr:`cond_cube`).
        """
        x_proj_full = self.manifold.projx(x)
        x_attended = torch.where(attn_mask.unsqueeze(-1), x_proj_full, x)
        if not self.no_detach:
            x_attended = x_attended.detach()
        if self.cond_cube:
            x_surr = x_attended.clone()
            in_p = _F(x_surr).in_p
            in_p.copy_(self.geom_trafos.to_cube(in_p))
        else:
            x_surr = x_attended
        return x_proj_full, x_attended, x_surr

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        attn_mask: torch.Tensor,
        types: torch.Tensor,
        pdgids: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Velocity at every slot, projected to the manifold tangent.

        Holds conditioning slots at ``t=1`` (zero path velocity) and
        projects the field onto the tangent space at attended slots.

        Args:
            x: features ``(B, L, in_dim)``.
            t: flow time, broadcastable to ``(B, L)``.
            mask: token roles ``(B, L)``; ``1`` = generated.
            attn_mask: real-token mask ``(B, L)``.
            types: slot-type ids ``(B, L)``.
            pdgids: species indices ``(B, L)``.
            d: step size for a step-conditioned (one-step-Euler) field,
                broadcastable to ``(B, L)``. Ignored unless the wrapped
                field has ``step_cond=True``; defaults to ``0`` (the
                instantaneous velocity slice) when omitted.

        Returns:
            Tangent velocity ``(B, L, in_dim)``.
        """
        x_proj_dense, _, x_surr = self._prep_x(x, attn_mask)
        pm = attn_mask.unsqueeze(-1)
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t = torch.where(mask == 1, t, 1.)
        if getattr(self.vf, "step_cond", False):
            d = x.new_zeros(()) if d is None else d
            d = torch.atleast_2d(d).expand_as(attn_mask)
        v = self.vf(x_surr, mask, attn_mask, types, pdgids, t=t, d=d)
        v_proj_dense = self.manifold.proju(x_proj_dense, v)
        return torch.where(pm, v_proj_dense, v)

class LEGOLtng(ltng.LightningModule):
    """Riemannian flow-matching model over padded particle sequences.

    Pairs a :class:`ProjectModel` velocity field with a manifold path
    sampler. Generation is mask-driven: slots with ``mask==0`` are held at
    their conditioning value, slots with ``mask==1`` are flowed from the
    base prior.
    """

    def __init__(self, full_config: dict) -> None:
        """Builds the model, base sampler, optimizer, and buffers.

        Args:
            full_config: raw config dict, resolved via
                :func:`~legofmt.mod_comps.config.resolve_legoltng_config`. A
                ``"state_dict"`` key triggers checkpoint restore, loaded
                non-strictly for backward compatibility.
        """
        super().__init__()
        rc = resolve_legoltng_config(full_config)
        self.rc = rc

        self.register_buffer("pdgids_template", rc.pdgids_template)
        self.register_buffer(
            "types_embd",
            torch.arange(rc.max_seq_l, dtype=torch.int64).clamp_max(3).view(1, -1),
        )

        self.model = self._build_model(rc)
        self.gen_base = GenerateBase(rc.config)
        self.ppa = CubeTrace()
        self.val_metrics = ShowerValMetrics()

        self.opt, self._lr_sched = build_optimizer(
            self.model.parameters(), rc.opt_conf,
        )
        self._opt_is_sf = hasattr(self.opt, "train") and callable(
            getattr(self.opt, "train", None),
        )

        if rc.state_dict is not None:
            self.model.vf.load_state_dict(rc.state_dict, strict=False)

    def _build_model(self, rc) -> nn.Module:
        """Constructs the wrapped velocity field; overridden by subclasses."""
        return ProjectModel(
            CFMTrafo_x(**rc.model_args),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )

    def _opt_train(self) -> None:
        """No-op unless the optimizer has a schedule-free .train() method."""
        if getattr(self, "_opt_is_sf", False):
            self.opt.train()

    def _opt_eval(self) -> None:
        """No-op unless the optimizer has a schedule-free .eval() method."""
        if getattr(self, "_opt_is_sf", False):
            self.opt.eval()

    @torch.no_grad()
    def on_fit_start(self) -> None:
        """Initialises the loss, path sampler, and train-mode optimizer."""
        if self.rc.ot_coupling and slap is None:
            raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
        self.loss_fn = nn.MSELoss()
        self.ps = ProductPathSampler(self.rc.manifold)
        self.model.train()
        self._opt_train()

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        """Maps raw PDG ids to compact indices into :attr:`pdgids_template`.

        Unknown, zero, or sentinel ids collapse to index ``0`` (the
        reserved empty/unknown class).

        Args:
            pdgids: raw PDG id tensor.

        Returns:
            Integer index tensor of the same shape.
        """
        cond = torch.isnan(pdgids) | (pdgids == 0) | (pdgids >= 1e8)
        pdgid_idx = torch.searchsorted(self.pdgids_template, pdgids.contiguous()) + 1
        return pdgid_idx.masked_fill_(cond, 0)

    @torch.no_grad()
    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]") -> Tensor:
        """Builds the flow's starting point: noise at generated slots, data elsewhere.

        Forward events (incoming slot conditions) draw the conditioning-aware
        ``poles`` prior around the incoming ray, OT-coupled to the targets
        during training when ``ot_coupling`` is set; inverse events draw the
        isotropic prior. Conditioning slots (``mask==0``) keep their data
        value, so the path interpolates base->data only where ``mask==1``.

        Returns:
            Base features ``(B, L, in_dim)``.
        """
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        data = ds_t.f.model_in
        m = ds_t.m.full
        fwd = m[:, 2] == 0
        noise = None if fwd.all() else self.gen_base.iso(m.shape, data.device)
        if fwd.any():
            base = torch.cat(
                (ds_t.f.non_cc, self.gen_base(ds_t.m.out_p.shape, ds_t.f.in_cc)), dim=1,
            )
            if self.rc.ot_coupling and self.model.training:
                if slap is None:
                    raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
                base = base.where(ds_t.am.full.unsqueeze(-1), data)
                inf_cond = ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
                out = _F(base).out_p
                if self.rc.ot_e_only:
                    nt = ds_t.f.out_cc[..., 0:1]
                    nb = out[..., 0].unsqueeze(-2)
                    cost = (nt - nb).abs() + inf_cond * 1e6
                else:
                    cost = torch.cdist(ds_t.f.out_cc, out) + inf_cond * 1e6
                assign = slap(cost, cost.device).long()
                out[:] = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
            base = self.gen_base.insert_add(base)
            noise = base if noise is None else torch.where(fwd.view(-1, 1, 1), base, noise)
        return torch.where((m == 1).unsqueeze(-1), noise, data)

    def _reduce_and_log(
        self,
        sq: Tensor,
        ds_t: DataStruct,
        loss_sc,
    ) -> Tensor:
        """Masked-mean of the per-component squared error, logged and summed.

        Averages the energy, direction, and position errors over the
        generated-and-real slots (``mask==1 & attn_mask==1``) so the scale
        is independent of the mask fraction, then adds the scaled auxiliary
        term.

        Args:
            sq: per-element squared error ``(B, L, in_dim)``.
            ds_t: the masked data struct for this step.
            loss_sc: auxiliary (energy) loss term, or ``0``.

        Returns:
            Scalar training loss.
        """
        g = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
        denom = g.sum().clamp(min=1)
        out = sq * g
        loss_e = out[..., 0:1].sum() / denom
        loss_dir = out[..., 1:4].sum() / (denom * 3)
        loss_x = out[..., 4:7].sum() / (denom * 3)
        if self.training:
            log_sc = loss_sc.detach() if torch.is_tensor(loss_sc) else loss_sc
            self.log_dict(
                {
                    "loss/energy": loss_e.detach(),
                    "loss/out_dir": loss_dir.detach(),
                    "loss/out_pos": loss_x.detach(),
                    "loss/sc": log_sc,
                },
                on_step=True, on_epoch=False, logger=True, sync_dist=False,
            )
        return loss_e + loss_dir + loss_x + self.rc.loss_sc_fac * loss_sc

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        """Samples the per-slot generation mask for a training step.

        Returns the dataset mask unchanged when no ``mask_conf`` is set
        (forward-only, backward-compatible). Otherwise each event keeps the
        dataset's forward mask with probability ``p_forward`` or flips to
        the inverse mask -- the complement of the forward mask over the
        attended slots, with the density slot always conditioning.

        Returns:
            Long mask ``(B, L)``; ``1`` = generated.
        """
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, 0] = 0
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv.long())

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        """One train/val step: the flow-matching MSE over a sampled mask
        and flow time.

        Returns:
            Scalar loss.
        """
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            base = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.rc.t_dist == "sm_norm":
                t = torch.sigmoid(self.rc.t_dist_scale * torch.randn_like(ds_t.f.d))
            elif self.rc.t_dist == "sd3":
                u = torch.rand_like(ds_t.f.d)
                t = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin()**2 - u)
            elif self.rc.t_dist == "sd3_grid":
                u = torch.rand_like(ds_t.f.d)
                t_sd3 = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin()**2 - u)
                idx = torch.multinomial(u.new_tensor([.1, .2, .3, .4]), u.numel(), replacement=True).view_as(u)
                t_grid = (u.new_tensor([0., 0.4, 0.8, 0.9])[idx] + 0.02 * torch.randn_like(u)).clamp(0, 1)
                t = torch.where(torch.rand_like(u) < 0.5, t_grid, t_sd3)
            elif self.rc.t_dist == "uniform":
                t = torch.rand_like(ds_t.f.d)
            ps_ = self.ps.sample(base, ds_t.f.model_in, t)
        v_out = self.model(
            ps_.x_t, ps_.t,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx,
        )
        if self.rc.loss_sc_fac > 0:
            am = ds_t.am.full
            pred = ((1 - ps_.t)[..., None] * v_out[..., 0:1] + ps_.x_t[..., 0:1]).squeeze(-1)
            loss_sc = self.loss_fn(pred * am, ds_t.f.energy.squeeze(-1) * am)
        else:
            loss_sc = 0.0
        sq = (v_out - ps_.dx_t) ** 2
        loss = self._reduce_and_log(sq, ds_t, loss_sc)

        if self.rc.one_step_euler_fac > 0:
            loss = loss + self.rc.one_step_euler_fac * self._one_step_euler_loss(base, ds_t, pdgid_idx, ps_)

        if self.rc.proj_dist_fac > 0:        # Idea 2: + distance after projecting to the sphere
            nrm = torch.nn.functional.normalize
            end = ps_.x_t + (1 - ps_.t)[..., None] * v_out
            pe = torch.cat([nrm(end[..., 1:4], dim=-1), nrm(end[..., 4:7], dim=-1)], -1)
            g = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
            proj = ((pe - ds_t.f.model_in[..., 1:7]) ** 2 * g).sum() / (g.sum().clamp(min=1) * 6)
            if self.training:
                self.log("loss/proj_dist", proj.detach(), on_step=True, on_epoch=False, logger=True)
            loss = loss + self.rc.proj_dist_fac * proj
        return loss

    def _one_step_euler_loss(
        self, base: Tensor, ds_t: DataStruct, pdgid_idx: Tensor, ps_,
    ) -> Tensor:
        mask, am = ds_t.m.full, ds_t.am.full
        gen = (mask == 1).unsqueeze(-1)
        g = gen & am.unsqueeze(-1)
        ckw = dict(mask=mask, attn_mask=am, types=self.types_embd, pdgids=pdgid_idx)
        man = self.model.manifold

        with torch.no_grad():
            i = torch.randint(1, self.rc.one_step_euler_sections + 1, (base.shape[0], 1), device=base.device)
            step = 2.0 ** -(i - 1).to(base.dtype)             # queried step D, (B, 1)
            half = step / 2
            t0 = (torch.rand_like(step) * (1.0 / step).round()).floor() * step  # grid-aligned start
            x0 = self.ps.sample(base, ds_t.f.model_in, t0.squeeze(-1)).x_t
            s_half = self.model(x0, t0, d=half, **ckw)
            x_mid = torch.where(gen, man.expmap(x0, half.unsqueeze(-1) * s_half), x0)
            s_mid = self.model(x_mid, t0 + half, d=half, **ckw)
            x_end = torch.where(gen, man.expmap(x_mid, half.unsqueeze(-1) * s_mid), x0)
            tgt = man.logmap(x0, x_end) / step.unsqueeze(-1)
        s_pred = self.model(x0, t0, d=step, **ckw)
        sc = ((s_pred - tgt) ** 2 * g).sum() / (g.sum().clamp(min=1) * s_pred.shape[-1])
        if self.training:
            self.log("loss/one_step_euler", sc.detach(), on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return sc

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        """Lightning training hook; delegates to :meth:`_step`."""
        loss = self._step(batch, _batch_idx)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        """Lightning validation hook; logs the loss and shower metrics."""
        bs = len(batch)
        loss = self._step(batch, _batch_idx)
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        for name, val in self.val_metrics(self, batch).items():
            self.log(name, val, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        """Returns the optimizer, with its LR scheduler when one is configured."""
        if self._lr_sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._lr_sched}

    def setup(self, stage: str | None = None) -> None:
        """Load the dataset once and carve off a held-out validation split."""
        if getattr(self, "_val_ds", None) is not None:
            return
        full = LEGODataset(**self.rc.dl_conf["lds_args"])
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=gen,
        )

    def _make_loader(self, dataset, *, shuffle: bool) -> DataLoader:
        """Builds a :class:`~torch.utils.data.DataLoader` over ``dataset``."""
        num_workers = self.rc.dl_conf.get("num_workers", 4)
        return DataLoader(
            dataset,
            batch_size=self.rc.dl_conf.get("bs", 2**12),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled loader over the training split."""
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Unshuffled loader over the validation split."""
        return self._make_loader(self._val_ds, shuffle=False)

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None):
        """Applies ``fn`` to ``tensors`` in row chunks, concatenating results.

        Runs ``fn`` once when ``split_size`` is unset or covers the whole
        batch; otherwise splits each tensor along ``dim`` and concatenates
        the outputs along ``cat_dim`` (defaulting to ``dim``). Bounds peak
        memory during sampling.
        """

        if split_size is None or split_size >= tensors[0].shape[dim]:
            return fn(*tensors)
        if cat_dim is None:
            cat_dim = dim
        out = []
        for chunk in zip(*(t.split(split_size, dim) for t in tensors)):
            out.append(fn(*chunk))
        return torch.cat(out, dim=cat_dim)

    def _prep_solve(
        self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
    ) -> tuple[DataStruct, Tensor]:
        """Switches to eval mode and resolves species indices before sampling.

        Returns:
            tuple: ``(ds_t, pdgids_idx)``.
        """
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if self.model.training:
            self.model.eval()
            self._opt_eval()
        pdgids = ds_t.f.pdgids
        pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
        return ds_t, pdgids_idx

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        reverse: bool = False,
        compute_ll: bool = False,
        log_p0=None,
        split_size: int | None = None,
        step_size: float = 0.04,
        method: str = "midpoint",
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        """Integrates the ODE from the base prior to particles (or its inverse).

        Wraps :class:`~flow_matching.solver.ODESolver`. With ``compute_ll``
        it returns the log-likelihood under the flow instead of samples;
        with ``reverse`` it integrates data->base. Conditioning slots stay
        pinned to their data value throughout.

        Args:
            ds_t: data struct (or ``(f, m, am)`` tuple) carrying the mask.
            x_init: starting features; defaults to the base prior.
            reverse: integrate ``t: 1->0`` instead of ``0->1``.
            compute_ll: return the log-likelihood rather than samples.
            log_p0: base log-density for the likelihood path.
            split_size: row-chunk size for bounded memory.
            step_size, method, time_grid: ODE-solver controls.
            return_intermediates: also return every solver step.

        Returns:
            Sampled features, log-likelihood, or the step stack per the flags.
        """
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        cc = ds_t.f.model_in.where(am, ds_t.f.in_cc)
        if x_init is not None and x_init.shape == cc.shape:
            x_init = x_init.where(am, _F(x_init).in_p)

        if method == "rk4":
            def vm(*args, **kwargs):
                return self.model(*args, **kwargs).clone()
        else:
            vm = self.model
        solver = ODESolver(velocity_model=vm)
        common = dict(step_size=step_size, method=method, types=self.types_embd)

        if compute_ll:
            self.model.no_detach = True
            try:
                _, log_ll = solver.compute_likelihood(
                    x_1=cc, log_p0=log_p0,
                    mask=ds_t.m.full, attn_mask=ds_t.am.full,
                    pdgids=pdgids_idx, **common,
                )
            finally:
                self.model.no_detach = False
            return log_ll

        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        explicit_grid = time_grid is not None
        if time_grid is None:
            if method == "euler":
                n = max(round(1.0 / step_size), 1)
                time_grid = torch.linspace(0., 1., n + 1, device=x_init.device, dtype=x_init.dtype)
                if reverse:
                    time_grid = time_grid.flip(0)
            else:
                time_grid = x_init.new_tensor([1.0, 0.0] if reverse else [0.0, 1.0])

        if method == "midpoint" and not explicit_grid:
            time_grid = x_init.new_tensor([1., 0.5, 0.] if reverse else [0., 0.5, 1.])

        def _sample(x_init, mask, attn_mask, pdgids_idx):
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd, pdgids=pdgids_idx)
            if method == "midpoint":
                return self._midpoint_steps(x_init, time_grid, **extras)
            if method == "euler":
                return self._euler_steps(
                    x_init, time_grid, return_intermediates=return_intermediates, **extras,
                )
            return solver.sample(
                x_init=x_init, time_grid=time_grid,
                mask=mask, attn_mask=attn_mask, pdgids=pdgids_idx,
                return_intermediates=return_intermediates, **common,
            )

        out = self.chunked(
            _sample, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx,
            split_size=split_size, cat_dim=-3,
        )
        if self.rc.proj_dist_fac > 0:
            nrm = torch.nn.functional.normalize
            out = torch.cat([out[..., 0:1], nrm(out[..., 1:4], dim=-1), nrm(out[..., 4:7], dim=-1)], -1)
        return out

    def _midpoint_steps(
        self, x: Tensor, time_grid: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> Tensor:
        """Fixed-grid midpoint (RK2) integration over ``time_grid``.

        Returns the final state, or -- when ``return_intermediates`` -- the
        stack of states at every ``time_grid`` point (``xs[0]`` is ``x_init``,
        matching :meth:`ODESolver.sample`).
        """
        if return_intermediates:
            xs = [x]
        man = self.model.manifold
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt = t_b - t_a
            v1 = self.model(x, t_a, **extras)
            x_half = man.expmap(x, dt / 2 * v1)
            v2 = self.model(x_half, t_a + dt / 2, **extras)
            x = man.expmap(x, dt * man.proju(x, v2))
            if return_intermediates:
                xs.append(x)
        return torch.stack(xs) if return_intermediates else x

    def _euler_steps(
        self, x: Tensor, time_grid: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> Tensor:
        gen = (extras["mask"] == 1).unsqueeze(-1)
        if return_intermediates:
            xs = [x]
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt = t_b - t_a
            s = self.model(x, t_a, d=dt.abs(), **extras)
            x = torch.where(gen, self.model.manifold.expmap(x, dt * s), x)
            if return_intermediates:
                xs.append(x)
        return torch.stack(xs) if return_intermediates else x

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        """Generates particles for ``batch`` end to end.

        Flows the kinematics with :meth:`solve` for the slots selected by
        the task mask. Non-attended slots are filled with ``NaN``.

        Args:
            batch: data struct (or ``(f, m, am)`` tuple) with the task mask.

        Returns:
            tuple: ``(features-with-pdgid, mask, attn_mask)``; the features
            gain a trailing PDG-id column and lead with the solver-step axis
            when intermediates are requested.
        """
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        cfg = self.rc.odeint_conf
        if (cfg.get("fwd_compile", False)
        and not (hasattr(self.model, "_orig_mod")
        or hasattr(self.model.vf, "_orig_mod"))):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        base = self.gen_base_wrapper(ds_t)

        pdgids = ds_t.f.pdgids
        am = ds_t.am.full.unsqueeze(-1)

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am, torch.nan)
        else:
            step_size = cfg.get("step_size", 0.04)
            time_grid = cfg.get("time_grid")
            if time_grid is None:
                time_grid = torch.arange(
                    0, 1 + step_size, step=step_size, device=self.device
                ).clamp_max(1)
            sols = self.solve(
                ds_t, x_init=base,
                split_size=cfg.get("split_size"),
                step_size=step_size,
                method=cfg.get("method", "midpoint"),
                time_grid=time_grid,
                return_intermediates=cfg.get("return_timesteps", False),
            )
            sols = sols.masked_fill_(~am, torch.nan)
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
                keep = torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid)) | (pdgids_idx == 0)
                sols.masked_fill_(~keep, torch.nan)
                pdgids = pdgids.masked_fill(~keep, 0)

        if sols.dim() == 4:
            T = sols.shape[0]
            pdgids = pdgids.unsqueeze(0).expand(T, -1, -1, -1)
        return torch.cat((sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full


def _build_reflow_teacher(reflow_path: str | None) -> nn.Module | None:
    """Frozen, ``torch.compile``'d velocity teacher for reflow; ``None`` if path unset/missing."""
    if reflow_path is None:
        return None
    if not Path(reflow_path).is_file():
        warnings.warn(f"reflow_path={reflow_path!r} not found; reflow disabled.", stacklevel=2)
        return None
    teacher = LEGOLtng(torch.load(reflow_path, map_location="cpu", weights_only=False))
    teacher.eval().requires_grad_(False)
    teacher.model = torch.compile(teacher.model, dynamic=False)
    return teacher


class ProjectModelDirect(ProjectModel):
    """Residual-prediction wrapper for the no-time direct model. Mirror
    of :class:`ProjectModel`; only the forward differs (no time argument,
    residual instead of velocity, single Euler step + safe sphere snap)."""

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_mask: torch.Tensor,
        types: torch.Tensor,
        pdgids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Residual step ``base + vf(base)`` with a manifold snap at generated slots.

        Returns:
            Final features ``(B, L, in_dim)``.
        """
        _, x_attended, x_surr = self._prep_x(x, attn_mask)
        residual = self.vf(x_surr, mask, attn_mask, types, pdgids)
        out_raw = x_attended + residual
        gen = (mask == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., 1] = 1.0
        ref[..., 4] = 1.0
        safe = torch.where(gen, out_raw, ref)
        out_proj = self.manifold.projx(safe)
        return torch.where(gen, out_proj, out_raw)


class LEGOLtngDirect(LEGOLtng):
    """Direct (residual-prediction) variant of :class:`LEGOLtng`.

    The transformer predicts the residual ``target - base`` at the
    generated slots; the wrapper applies one Euler step
    ``final = base + residual`` (plus a manifold snap on the sphere half)
    and the loss is MSE between predicted and target.

    When ``model_conf.reflow_path`` is set to a velocity-model checkpoint,
    that model is loaded as a frozen teacher and ``teacher.solve(base)``
    replaces the data target — giving the direct student a fixed
    ``base -> target`` coupling per batch (the prerequisite for one-step
    Euler to reach the target). When the path is unset or missing,
    training falls back to MSE against the data target.
    """

    def __init__(self, full_config: dict) -> None:
        """Builds the direct model and, if configured, the frozen reflow teacher."""
        super().__init__(full_config)
        object.__setattr__(
            self, "reflow_teacher", _build_reflow_teacher(self.rc.reflow_path),
        )

    def _build_model(self, rc) -> nn.Module:
        """Constructs the no-time residual-prediction wrapper."""
        return ProjectModelDirect(
            CFMTrafo_x(**rc.model_args, time_cond=False),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )

    @torch.no_grad()
    def on_fit_start(self) -> None:
        """Initialises base state and moves the reflow teacher to the device."""
        super().on_fit_start()
        if self.reflow_teacher is not None:
            self.reflow_teacher.to(self.device)

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        """One direct-model step: residual MSE to the target (data, or the
        reflow teacher's solved coupling).

        Returns:
            Scalar loss.
        """
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            base = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.reflow_teacher is not None:
                solve_kwargs = dict(self.rc.reflow_kwargs)
                if (
                    "time_grid" not in solve_kwargs
                    and solve_kwargs.get("method", "midpoint") == "midpoint"
                ):
                    solve_kwargs["time_grid"] = base.new_tensor([0.0, 1.0])
                target = self.reflow_teacher.solve(
                    ds_t, x_init=base, **solve_kwargs,
                )
            else:
                target = ds_t.f.model_in
        pred = self.model(
            base,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx,
        )
        if self.rc.loss_sc_fac > 0:
            m_gen = (ds_t.m.full == 1).to(pred.dtype)
            loss_sc = self.loss_fn(
                pred[..., 0] * m_gen,
                target[..., 0] * m_gen,
            )
        else:
            loss_sc = 0.0
        sq = (pred - target) ** 2
        return self._reduce_and_log(sq, ds_t, loss_sc)

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        split_size: int | None = None,
        **_kw,
    ) -> Tensor:
        """Single-pass residual prediction from the base prior (no ODE)."""
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        def _fwd(x, m, a, pi):
            return self.model(
                x, mask=m, attn_mask=a, types=self.types_embd, pdgids=pi,
            )

        return self.chunked(
            _fwd, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx,
            split_size=split_size,
        )
