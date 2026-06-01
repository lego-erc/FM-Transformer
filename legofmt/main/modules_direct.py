"""Direct (residual-prediction) mirror of :class:`legofmt.main.modules.LEGOLtng`.

The transformer predicts the residual ``target - base`` at the generated
slots; the wrapper applies one Euler step ``final = base + residual``
(plus a manifold snap on the sphere half) and returns the predicted
target. The loss is MSE between that predicted target and the per-batch
training target.

When ``model_conf.reflow_path`` is set to a velocity-model checkpoint,
this module loads it as a frozen teacher and uses ``teacher.solve(base)``
as the training target. That replaces the data target with the velocity
model's deterministic ODE map, giving the direct student a fixed
``base -> target`` coupling per batch (the prerequisite for one-step
Euler to reach the target). When the path is unset or missing, training
falls back to ``MSE(pred, data target)`` as before.
"""

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

import lightning as ltng

try:
    from torch_lap_cuda_lib import solve_lap as slap
except ImportError:
    slap = None

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.struct import DataStruct, _F
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.cfm.cfm_trafo_direct import CFMTrafo_x
from legofmt.geometry.gen_base import GenerateBase
from legofmt.geometry.path_sample_mult import ProductManifold
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer
from legofmt.mod_comps.val_metrics import ShowerValMetrics


class ProjectModel(nn.Module):
    """Residual-prediction wrapper for the no-time direct model.

    The transformer predicts the offset ``target - base`` at gen slots; the
    wrapper applies one Euler step ``final = base + residual`` and snaps the
    sphere half back to the manifold, returning the predicted target. Mirror
    of :class:`legofmt.main.modules.ProjectModel`; the velocity version
    returns a tangent velocity via ``proju``.
    """

    def __init__(
        self,
        vf: nn.Module,
        manifold: ProductManifold,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.vf = vf
        self.manifold = manifold
        self.geom_trafos = GeomTrafos()
        self.cond_cube = kwargs.get("cond_cube", False)
        self.no_detach = kwargs.get("no_detach", False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_mask: torch.Tensor,
        types: torch.Tensor,
        pdgids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pm = attn_mask.unsqueeze(-1)
        x_proj_dense = self.manifold.projx(x)
        x = torch.where(pm, x_proj_dense, x)
        if not self.no_detach:
            x = x.detach()
        if self.cond_cube:
            x_cube = x.clone()
            in_p = _F(x_cube).in_p
            in_p.copy_(self.geom_trafos.to_cube(in_p))
            x_surr = x_cube
        else:
            x_surr = x
        residual = self.vf(x_surr, mask, attn_mask, types, pdgids)
        out_raw = x + residual
        gen = (mask == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., 3] = 1.0
        safe = torch.where(gen, out_raw, ref)
        out_proj = self.manifold.projx(safe)
        return torch.where(gen, out_proj, out_raw)


class LEGOLtng(ltng.LightningModule):
    def __init__(self, full_config: dict) -> None:
        super().__init__()
        rc = resolve_legoltng_config(full_config)
        self.rc = rc

        self.register_buffer("pdgids_template", rc.pdgids_template)
        self.register_buffer(
            "types_embd",
            torch.arange(rc.max_seq_l, dtype=torch.int64).clamp_max(3).view(1, -1),
        )

        self.model = ProjectModel(
            CFMTrafo_x(**rc.model_args),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )
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

        teacher = self._maybe_build_reflow_teacher(rc.reflow_path)
        self._reflow_teacher_box = [teacher] if teacher is not None else []

    @property
    def reflow_teacher(self) -> nn.Module | None:
        return self._reflow_teacher_box[0] if self._reflow_teacher_box else None

    @staticmethod
    def _maybe_build_reflow_teacher(reflow_path: str | None) -> nn.Module | None:
        """Load the velocity-model checkpoint at ``reflow_path`` as a frozen
        teacher. Returns ``None`` if the path is unset or the file is missing
        (the latter case keeps saved-then-relocated direct checkpoints loadable
        without the teacher file).

        The teacher's wrapper is ``torch.compile``'d so its 4 forwards per
        midpoint step run at compiled throughput (matching the student's
        compiled forward). Without this the teacher dominates each step.
        """
        if reflow_path is None:
            return None
        if not Path(reflow_path).is_file():
            import warnings
            warnings.warn(
                f"reflow_path={reflow_path!r} not found; reflow disabled. "
                "Training will use the data target. Set to ``None`` in the "
                "saved config to silence.",
                stacklevel=2,
            )
            return None
        from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity
        teacher_ckpt = torch.load(reflow_path, map_location="cpu", weights_only=False)
        teacher = LEGOLtngVelocity(teacher_ckpt)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        try:
            teacher.model = torch.compile(teacher.model, dynamic=False)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"torch.compile of reflow teacher failed ({exc!r}); "
                "falling back to eager. Each training step will run 4 eager "
                "teacher forwards, which is the dominant cost.",
                stacklevel=2,
            )
        return teacher

    def _opt_train(self) -> None:
        if self._opt_is_sf:
            self.opt.train()

    def _opt_eval(self) -> None:
        if self._opt_is_sf:
            self.opt.eval()

    @torch.no_grad()
    def on_fit_start(self) -> None:
        if self.rc.ot_coupling and slap is None:
            raise RuntimeError(
                "ot_coupling=True requires `torch_lap_cuda_lib`. "
                "Install it or set model_conf.ot_coupling=False."
            )
        self.loss_fn = nn.MSELoss()
        if self.reflow_teacher is not None:
            self.reflow_teacher.to(self.device)
        self.model.train()
        self._opt_train()

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        cond = torch.isnan(pdgids) | (pdgids == 0) | (pdgids >= 1e8)
        pdgid_idx = torch.searchsorted(self.pdgids_template, pdgids.contiguous()) + 1
        return pdgid_idx.masked_fill_(cond, 0)

    @torch.no_grad()
    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]") -> Tensor:
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        base = torch.cat((ds_t.f.non_cc, self.gen_base(ds_t.m.out_p.shape, ds_t.f.in_cc)), dim=1)
        if (
            self.rc.ot_coupling
            and self.model.training
            and self.reflow_teacher is None
        ):
            base = base.where(ds_t.am.full.unsqueeze(-1), ds_t.f.model_in)
            inf_cond = ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
            out = _F(base).out_p
            if self.rc.ot_e_only:
                nt = ds_t.f.out_cc[..., :3].norm(dim=-1)[..., None]
                nb = out[..., :3].norm(dim=-1)[..., None, :]
                cost = (nt - nb).abs() + inf_cond * 1e6
            else:
                cost = torch.cdist(ds_t.f.out_cc, out) + inf_cond * 1e6
            assign = slap(cost, cost.device).long()
            out[:] = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
        return self.gen_base.insert_add(base)

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
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

        sq = (pred - target) ** 2
        gen = (ds_t.m.out_p.unsqueeze(-1) == 1)
        out_sq = sq[:, 3:] * gen
        loss_edep = sq[:, 1].mean()
        loss_p = out_sq[..., :3].mean()
        loss_x = out_sq[..., 3:].mean()
        if self.rc.loss_sc_fac > 0:
            m_gen = (ds_t.m.full == 1).to(pred.dtype)
            pred_mag = pred[..., :3].norm(dim=-1) * m_gen
            tgt_mag = target[..., :3].norm(dim=-1) * m_gen
            loss_sc = self.loss_fn(pred_mag, tgt_mag)
        else:
            loss_sc = pred.new_zeros(())

        if self.training:
            self.log_dict(
                {
                    "loss/edep": loss_edep.detach(),
                    "loss/out_eucl": loss_p.detach(),
                    "loss/out_sph": loss_x.detach(),
                    "loss/sc": loss_sc.detach(),
                },
                on_step=True, on_epoch=False, logger=True, sync_dist=False,
            )
        return loss_edep + loss_p + loss_x + self.rc.loss_sc_fac * loss_sc

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
        full = LEGODataset(**self.rc.dl_conf["lds_args"])
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=gen,
        )

    def _make_loader(self, dataset, *, shuffle: bool) -> DataLoader:
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
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        split_size: int | None = None,
        **_kw,
    ) -> Tensor:
        """Single forward pass; the wrapper performs the Euler step
        ``final = base + residual`` and the manifold snap internally.

        ``split_size`` chunks the batch dimension when running into memory
        pressure with very large inference batches. Pick a value that
        divides the batch size to avoid recompile on the partial last chunk
        when the wrapper is ``torch.compile``'d with ``dynamic=False``."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        pdgids = ds_t.f.pdgids
        pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)

        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        mask, attn_mask = ds_t.m.full, ds_t.am.full
        if split_size is None or split_size >= x_init.shape[0]:
            return self.model(
                x_init,
                mask=mask, attn_mask=attn_mask,
                types=self.types_embd, pdgids=pdgids_idx,
            )
        parts = []
        for i in range(0, x_init.shape[0], split_size):
            sl = slice(i, i + split_size)
            parts.append(self.model(
                x_init[sl],
                mask=mask[sl], attn_mask=attn_mask[sl],
                types=self.types_embd, pdgids=pdgids_idx[sl],
            ))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        cfg = self.rc.odeint_conf
        if (
            cfg.get("fwd_compile", False)
            and not (
                hasattr(self.model, "_orig_mod")
                or hasattr(self.model.vf, "_orig_mod")
            )
        ):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        pdgids = ds_t.f.pdgids
        am = ds_t.am.full.unsqueeze(-1)
        base = self.gen_base_wrapper(ds_t)
        densities = ds_t.f.d[:, None, None].expand_as(base[..., :1])

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am, torch.nan)
        else:
            sols = self.solve(ds_t, x_init=base, split_size=cfg.get("split_size"))
            sols = sols.masked_fill_(~am, torch.nan)
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
                keep = (
                    torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid))
                    | (pdgids_idx == 0)
                )
                sols.masked_fill_(~keep, torch.nan)
                pdgids = pdgids.masked_fill(~keep, 0)

        return torch.cat((densities, sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full
