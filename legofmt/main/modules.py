import contextlib

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
from legofmt.cfm.solvers import Solvers

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.prep import DataPrep
from legofmt.data.struct import DataStruct, _F

from legofmt.distill.distill import curvature_loss, one_step_euler_loss

from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.path_sample_mult import ProductPathSampler, ProductManifold
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.mod_comps.config import _amp_dtype, resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer

from legofmt.log_metrics.val_metrics import ShowerValMetrics


class ProjectModel(nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(self, vf: nn.Module, manifold: ProductManifold, **kwargs) -> None:
        super().__init__()
        self.vf          = vf
        self.manifold    = manifold
        self.geom_trafos = GeomTrafos()
        self.cond_cube   = kwargs.get("cond_cube", False)
        self.no_detach   = kwargs.get("no_detach", False)

    def _prep_x(self, x: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        x_proj = self.manifold.projx(x)
        x_att  = torch.where(attn_mask.unsqueeze(-1), x_proj, x)
        if not self.no_detach:
            x_att.detach_()
        if self.cond_cube:
            x_att = x_att.clone()
            in_p  = _F(x_att).in_p
            in_p.copy_(self.geom_trafos.to_cube(in_p))
        return x_proj, x_att

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        x_proj, x_att = self._prep_x(x, attn_mask)
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t = torch.where(mask == 1, t, 1.0)  # conditions get t=1
        if getattr(self.vf, "step_cond", False):
            d = torch.zeros_like(t) if d is None else torch.atleast_2d(d).expand_as(attn_mask)
            d = torch.where(mask == 1, d, 0.0)  # conditions are not transported
        v = self.vf(x_att, mask, attn_mask, types, pdgids, t=t, d=d)
        v_proj = self.manifold.proju(x_proj, v)
        return torch.where(attn_mask.unsqueeze(-1), v_proj, v)


class LEGOLtng(BaseDist, Solvers, ltng.LightningModule):

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

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        f = ds_t.f.full.clone()
        if self.rc.overflow_delta > 0:
            op = _F(f).out_p
            op[..., 0].masked_fill_((op[..., 0] == 0) & ds_t.am.out_p.bool(),
                                    -self.rc.overflow_delta)
        if self.rc.edep_overflow_delta > 0:
            edep = _F(f).edep
            edep.masked_fill_(edep == 0, -self.rc.edep_overflow_delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    def _canon_dirs(self, x: Tensor, face: Tensor, fwd: Tensor, inverse: bool = False) -> Tensor:
        rot  = self.sym.uncanonicalize if inverse else self.sym.canonicalize
        dirs = rot(torch.stack((x[..., 1:4], x[..., 4:7]), dim=-2), face)
        new  = torch.cat((x[..., 0:1], dirs[..., 0, :], dirs[..., 1, :], x[..., 7:]), dim=-1)
        rows = torch.arange(x.shape[-2], device=x.device) >= self.rc.n_prefix
        new  = torch.where(rows.view(1, -1, 1), new, x)
        return torch.where(fwd[:, None, None], new, x)

    def _reduce_and_log(
        self, sq: Tensor, ds_t: DataStruct, loss_sc, t: Tensor | None = None,
    ) -> Tensor:
        g        = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
        denom    = g.sum().clamp(min=1)
        out      = sq * g
        loss_e   = out[..., 0:1].sum() / denom
        loss_dir = out[..., 1:4].sum() / (denom * 3)
        loss_x   = out[..., 4:7].sum() / (denom * 3)
        logs     = {}
        if not self.rc.uncert_weighting:
            total = loss_e + loss_dir + loss_x
        else:
            if t is None:
                raise ValueError(
                    "uncert_weighting=True needs the per-event t, which the "
                    "time-independent (LEGOLtngDirect) path does not have."
                )
            nb  = self.rc.uncert_bins
            idx = (t.detach().clamp(0, 1) * nb).long().clamp(max=nb - 1)
            lv  = self.lv.view(nb, 3)[idx].clamp(min=self.rc.uncert_min)  # (B, 3)
            w   = torch.exp(-lv)
            total = (
                (out[..., 0:1] * w[:, 0].view(-1, 1, 1)).sum() / denom
                + (out[..., 1:4] * w[:, 1].view(-1, 1, 1)).sum() / (denom * 3)
                + (out[..., 4:7] * w[:, 2].view(-1, 1, 1)).sum() / (denom * 3)
                + lv.mean(0).sum()  # the +log(sigma^2) barrier; zero at init
            )
            v = lv.detach().exp().mean(0)
            logs = {"uncert/var_energy": v[0], "uncert/var_dir": v[1], "uncert/var_pos": v[2]}
        if self.training:
            log_sc = loss_sc.detach() if torch.is_tensor(loss_sc) else loss_sc
            self.log_dict(
                {
                    # unweighted, so these stay comparable across uncert settings
                    "loss/energy": loss_e.detach(),
                    "loss/out_dir": loss_dir.detach(),
                    "loss/out_pos": loss_x.detach(),
                    "loss/raw": (loss_e + loss_dir + loss_x).detach(),
                    "loss/sc": log_sc,
                    **logs,
                },
                on_step=True, on_epoch=False, logger=True, sync_dist=False,
            )
        return total + self.rc.loss_sc_fac * loss_sc

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, :self.rc.n_prefix] = 0  # all conditioning scalars stay conditioning
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv.long())

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            if self.rc.overflow_delta > 0 or self.rc.edep_overflow_delta > 0:
                ds_t = self._shift_overflow_targets(ds_t)
            if self.sym is not None:
                fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
                face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
                ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd), ds_t.m.full, ds_t.am.full)
            base      = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.rc.t_dist == "sm_norm":
                t = torch.sigmoid(self.rc.t_dist_scale * torch.randn_like(ds_t.f.d))
            elif self.rc.t_dist == "sd3":
                u = torch.rand_like(ds_t.f.d)
                t = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin() ** 2 - u)
            else:
                raise ValueError(f"unknown t_dist: {self.rc.t_dist!r}")
            if self.rc.t_zero_frac > 0:
                t = t * (torch.rand_like(t) >= self.rc.t_zero_frac)
            if self.rc.t_dist_shift != 1.0:
                t = t.clamp(min=0) ** (1.0 / self.rc.t_dist_shift)
            if (
                self.reflow_teacher is not None and self.training
                and self.global_step % self.rc.reflow_every == 0
            ):  # reflow: couple base to the teacher's transport of it
                tgt = self.reflow_teacher.solve(ds_t, x_init=base, **self.rc.reflow_kwargs)
                tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
                ds_t = DataStruct(
                    torch.cat((tgt, ds_t.f.pdgids), dim=-1), ds_t.m.full, ds_t.am.full,
                )
            ps_ = self.ps.sample(base, ds_t.f.model_in, t)
        step_extras = ({"d": torch.zeros_like(ps_.t)}
                       if getattr(self.model.vf, "step_cond", False) else {})
        v_out = self.model(
            ps_.x_t, ps_.t,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx, **step_extras,
        )
        if self.rc.loss_sc_fac > 0:
            am = ds_t.am.full
            pred = ((1 - ps_.t)[..., None] * v_out[..., 0:1] + ps_.x_t[..., 0:1]).squeeze(-1)
            loss_sc = self.loss_fn(pred * am, ds_t.f.energy.squeeze(-1) * am)
        else:
            loss_sc = 0.0
        sq = (v_out - ps_.dx_t) ** 2
        loss = self._reduce_and_log(sq, ds_t, loss_sc, t=t)

        every = self.rc.one_step_euler_every
        fac   = self.rc.one_step_euler_fac
        lvf   = None
        if fac > 0 and hasattr(self, "lv_flow"):
            # Barrier every step, data term only when the gate fires: the
            # expectations still meet at exp(lv_flow) = E[L_flow], and lv_flow
            # never leaves the graph (DDP find_unused_parameters=False).
            lvf  = self.lv_flow.clamp(min=self.rc.uncert_min).squeeze()
            loss = loss + fac * lvf
        if fac > 0 and (not self.training or self.global_step % every == 0):
            sc = one_step_euler_loss(self, base, ds_t, pdgid_idx)
            w = fac * (every if self.training else 1)
            if lvf is not None:
                w = w * torch.exp(-lvf)
            if self.training:
                logs = {"loss/one_step_euler": sc.detach()}
                if lvf is not None:
                    logs["uncert/var_flow"] = lvf.detach().exp()
                self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
            loss = loss + w * sc

        if self.rc.curv_fac > 0 and self.training and self.global_step % self.rc.curv_every == 0:
            w    = self.rc.curv_warmup
            ramp = 1.0 if w <= 0 else min(1.0, self.global_step / w)
            if ramp > 0:
                cv = curvature_loss(self, ps_.x_t, ps_.t, v_out, ds_t, pdgid_idx)
                self.log("loss/curvature", cv.detach(), on_step=True, on_epoch=False, logger=True, sync_dist=False)
                loss = loss + self.rc.curv_fac * self.rc.curv_every * ramp * cv

        if (ot := self._base_dist_loss) is not None:
            loss = loss + self.rc.base_dist_loss * ot
            if self.training:
                self.log_dict(
                    {"loss/base_dist": ot.detach(),
                     "base/sm_scale": self.gen_base.sm_scale.detach().mean()},
                    on_step=True, on_epoch=False, logger=True, sync_dist=False,
                )
        return loss

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        return self._step(batch, _batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        bs = len(batch)
        loss = self._step(batch, _batch_idx)
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        for name, val in self.val_metrics(self, batch).items():
            self.log(name, val, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

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

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        cfg = self.rc.odeint_conf
        if cfg.get("fwd_compile", False) and not (
            hasattr(self.model, "_orig_mod") or hasattr(self.model.vf, "_orig_mod")
        ):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        if self.sym is not None:
            fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
            face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd), ds_t.m.full, ds_t.am.full)
        base = self.gen_base_wrapper(ds_t)

        pdgids = ds_t.f.pdgids
        am     = ds_t.am.full.unsqueeze(-1)

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am, torch.nan)
        else:
            step_size = cfg.get("step_size", 0.04)
            time_grid = cfg.get("time_grid")
            if time_grid is None:
                time_grid = torch.arange(
                    0, 1 + step_size, step=step_size, device=self.device
                ).clamp_max(1)
            amp = (_amp_dtype(cfg["amp"]) if "amp" in cfg
                   else self.rc.amp_dtype)
            with (contextlib.nullcontext() if amp is None
                  else torch.autocast(base.device.type, dtype=amp)):
                sols = self.solve(
                    ds_t, x_init=base,
                    split_size=cfg.get("split_size"),
                    step_size=step_size,
                    method=cfg.get("method", "midpoint"),
                    time_grid=time_grid,
                    return_intermediates=cfg.get("return_timesteps", False),
                )
            sols = sols.float()
            sols = sols.masked_fill_(~am, torch.nan)
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
                keep = torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid)) | (pdgids_idx == 0)
                sols.masked_fill_(~keep, torch.nan)
                pdgids = pdgids.masked_fill(~keep, 0)

        if self.sym is not None:
            if sols.dim() == 4:
                sols = torch.stack([self._canon_dirs(s, face, fwd, inverse=True) for s in sols])
            else:
                sols = self._canon_dirs(sols, face, fwd, inverse=True)

        if sols.dim() == 4:
            pdgids = pdgids.unsqueeze(0).expand(sols.shape[0], -1, -1, -1)
        return torch.cat((sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full
