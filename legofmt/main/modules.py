import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import lightning as ltng

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

try:
    from torch_lap_cuda_lib import solve_lap as slap
except ImportError:
    slap = None

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.struct import DataStruct, _F
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.geometry.gen_base import GenerateBase
from legofmt.geometry.path_sample_mult import ProductPathSampler, ProductManifold
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.main.config import resolve_legoltng_config
from legofmt.main.optimizers import build_optimizer


class ProjectModel(ModelWrapper, nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(
        self,
        vf: nn.Module,
        manifold: ProductManifold,
        **kwargs: dict,
    ) -> None:
        nn.Module.__init__(self)
        self.vf = vf
        self.manifold = manifold
        self.kwargs = kwargs
        self.geom_trafos = GeomTrafos()
        self.cond_cube = kwargs.get("cond_cube", False)
        self.no_detach = kwargs.get("no_detach", False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
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
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t = torch.where(mask == 1, t, 1.)
        if self.cond_cube:
            x_cube = x.clone()
            in_p = _F(x_cube).in_p
            in_p.copy_(self.geom_trafos.to_cube(in_p))
            x_surr = x_cube
        else:
            x_surr = x
        v = self.vf(t, x_surr, mask, attn_mask, types, pdgids)
        v_proj_dense = self.manifold.proju(x_proj_dense, v)
        return torch.where(pm, v_proj_dense, v)


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

        self.opt, self._lr_sched = build_optimizer(
            self.model.parameters(), rc.opt_conf,
        )
        self._opt_is_sf = hasattr(self.opt, "train") and callable(
            getattr(self.opt, "train", None),
        )

        if rc.state_dict is not None:
            self.model.vf.load_state_dict(rc.state_dict, strict=False)

    def _opt_train(self) -> None:
        """No-op unless the optimizer has a schedule-free .train() method."""
        if self._opt_is_sf:
            self.opt.train()

    def _opt_eval(self) -> None:
        """No-op unless the optimizer has a schedule-free .eval() method."""
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
        self.ps = ProductPathSampler(self.rc.manifold)
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
        return self.gen_base.insert_add(base)  # E_dep

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
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
            pred = ((1 - ps_.t)[..., None] * v_out[..., :3] + ps_.x_t[..., :3]).norm(dim=-1)
            loss_sc = self.loss_fn(pred * am, ds_t.f.mom.norm(dim=-1) * am)
        else:
            loss_sc = 0.0
        sq = (v_out - ps_.dx_t)**2
        loss_v = sq[:, 1, 0].mean() + (sq[:, 3:] * (ds_t.m.out_p == 1).unsqueeze(-1)).mean()
        return loss_v + self.rc.loss_sc_fac * loss_sc

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        with torch.no_grad():
            self.log(
                "Validation Loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        return loss

    def configure_optimizers(self):
        if self._lr_sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._lr_sched}

    def train_dataloader(self) -> DataLoader:
        num_workers = self.rc.dl_conf.get("num_workers", 4)
        dataset_train = LEGODataset(**self.rc.dl_conf.get("lds_args"))
        return DataLoader(
            dataset_train,
            batch_size=self.rc.dl_conf.get("bs", 2**12),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None):

        if split_size is None or split_size >= tensors[0].shape[dim]:
            return fn(*tensors)
        if cat_dim is None:
            cat_dim = dim
        out = []
        for chunk in zip(*(t.split(split_size, dim) for t in tensors)):
            out.append(fn(*chunk))
        return torch.cat(out, dim=cat_dim)

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
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        pdgids = ds_t.f.pdgids
        pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
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
            time_grid = x_init.new_tensor([1.0, 0.0] if reverse else [0.0, 1.0])

        use_2step = (
            method == "midpoint" and step_size == 0.5 and not return_intermediates
        )
        if use_2step and not explicit_grid:
            time_grid = x_init.new_tensor([1., 0.5, 0.] if reverse else [0., 0.5, 1.])

        def _sample(x_init, mask, attn_mask, pdgids_idx):
            if use_2step:
                return self._midpoint_steps(
                    x_init, time_grid,
                    mask=mask, attn_mask=attn_mask,
                    types=self.types_embd, pdgids=pdgids_idx,
                )
            return solver.sample(
                x_init=x_init, time_grid=time_grid,
                mask=mask, attn_mask=attn_mask, pdgids=pdgids_idx,
                return_intermediates=return_intermediates, **common,
            )

        return self.chunked(
            _sample, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx,
            split_size=split_size, cat_dim=-3,
        )

    def _midpoint_steps(self, x: Tensor, time_grid: Tensor, **extras) -> Tensor:
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt = t_b - t_a
            v1 = self.model(x, t_a, **extras)
            v2 = self.model(x + dt / 2 * v1, t_a + dt / 2, **extras)
            x = x + dt * v2
        return x

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        cfg = self.rc.odeint_conf
        if (cfg.get("fwd_compile", False)
        and not (hasattr(self.model, "_orig_mod")
        or hasattr(self.model.vf, "_orig_mod"))):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        pdgids = ds_t.f.pdgids
        am = ds_t.am.full.unsqueeze(-1)
        base = self.gen_base_wrapper(ds_t)
        densities = ds_t.f.d[:, None, None].expand_as(base[..., :1])

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am, torch.nan)
        else:
            step_size = cfg.get("step_size", 0.04)
            time_grid = (
                torch.arange(
                    0, 1 + step_size, step=step_size, device=self.device
                ).clamp_max(1)
                if cfg.get("return_timesteps", False)
                else None
            )
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
            densities = densities.unsqueeze(0).expand(T, -1, -1, -1)
            pdgids = pdgids.unsqueeze(0).expand(T, -1, -1, -1)
        return torch.cat((densities, sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full
