"""Direct (residual-prediction) mirror of :class:`legofmt.main.modules.LEGOLtng`.

The transformer predicts the residual ``target - base`` at the generated
slots; the Euler step ``final = base + residual`` and the manifold snap
are applied in :meth:`LEGOLtng.solve` (a single step from t=0 to t=1).
The loss is MSE between the predicted residual and ``target - base``.
"""

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

    Returns the predicted residual ``target - base`` (gated to gen slots by
    the transformer's own mask). The Euler step ``final = base + residual``
    and the manifold snap live in :meth:`LEGOLtng.solve`. Mirror of
    :class:`legofmt.main.modules.ProjectModel`; the velocity version returns
    a tangent velocity via ``proju``, this one returns the residual raw.
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
        self.kwargs = kwargs
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
        # Residual ``target - base`` at gen slots (zeroed elsewhere by the
        # transformer's mask gate). No projx on the output: it is added to the
        # base downstream and snapped to the manifold there.
        return self.vf(x_surr, mask, attn_mask, types, pdgids)


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
        # OT coupling: tighten base<->target pairing within each batch so the
        # direct map is well-defined per-event instead of averaging across
        # randomly paired targets (which collapses to the conditional mean).
        if self.rc.ot_coupling and self.model.training:
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
            # Residual target: the model predicts ``target - base``; one Euler
            # step adds it back at inference (see :meth:`solve`).
            target = ds_t.f.model_in - base
        residual = self.model(
            base,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx,
        )

        sq = (residual - target) ** 2
        gen = (ds_t.m.out_p.unsqueeze(-1) == 1)
        out_sq = sq[:, 3:] * gen
        loss_edep = sq[:, 1].mean()
        loss_p = out_sq[..., :3].mean()
        loss_x = out_sq[..., 3:].mean()
        # Magnitude loss on the FINAL outgoing momentum ``base + residual``.
        # Per-component MSE averages to ~0 over isotropic targets; this term
        # gives a direction-independent magnitude signal. Set ``loss_sc`` ~1
        # for the direct model (the velocity-model default of 1e-3 is too
        # small here).
        if self.rc.loss_sc_fac > 0:
            m_gen = (ds_t.m.full == 1).to(residual.dtype)
            pred_mom = base[..., :3] + residual[..., :3]
            pred_mag = pred_mom.norm(dim=-1) * m_gen
            tgt_mag = ds_t.f.mom.norm(dim=-1) * m_gen
            loss_sc = self.loss_fn(pred_mag, tgt_mag)
        else:
            loss_sc = residual.new_zeros(())

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
        **_kw,
    ) -> Tensor:
        """Single Euler step from t=0 to t=1: ``final = base + residual`` with
        a manifold snap on the generated slots. Called by ``ShowerValMetrics``
        and :meth:`forward`."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        pdgids = ds_t.f.pdgids
        pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)

        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        residual = self.model(
            x_init,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgids_idx,
        )
        out_raw = x_init + residual
        gen = (ds_t.m.full == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., 3] = 1.0  # unit-norm position; mom half is Euclidean (no projx)
        safe = torch.where(gen, out_raw, ref)
        out_proj = self.model.manifold.projx(safe)
        return torch.where(gen, out_proj, out_raw)

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        pdgids = ds_t.f.pdgids
        am = ds_t.am.full.unsqueeze(-1)
        base = self.gen_base_wrapper(ds_t)
        densities = ds_t.f.d[:, None, None].expand_as(base[..., :1])

        sols = self.solve(ds_t, x_init=base)
        sols = sols.masked_fill(~am, torch.nan)
        return torch.cat((densities, sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full
