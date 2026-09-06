"""ODE solving / sampling for the Riemannian FM model.

Split out of ``main/modules.py``. These are mixin methods rather than a
standalone solver object on purpose: ``lego_eval`` version-probes them with
``hasattr(ltng, "_midpoint_steps")`` / ``hasattr(lego, "log_likelihood")``, so
they must stay bound on the LightningModule.

Members resolved through ``self`` and owned by ``LEGOLtng``: ``model``,
``types_embd``, ``rc``, ``gen_base_wrapper``, ``convert_pdgids``, ``_opt_eval``.
"""

import torch
from torch import Tensor

from flow_matching.solver import ODESolver

from legofmt.data.struct import DataStruct, _F


class Solvers:
    """Sampling / likelihood mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None):
        if split_size is None or split_size >= tensors[0].shape[dim]:
            return fn(*tensors)
        if cat_dim is None:
            cat_dim = dim
        out = [fn(*chunk) for chunk in zip(*(t.split(split_size, dim) for t in tensors))]
        return torch.cat(out, dim=cat_dim)

    def _prep_solve(
        self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
    ) -> tuple[DataStruct, Tensor]:
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if self.model.training:
            self.model.eval()
            self._opt_eval()
        pdgids     = ds_t.f.pdgids
        pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
        return ds_t, pdgids_idx

    @torch.no_grad()
    def log_likelihood(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        log_p0,
        step_size: float = 0.04,
        method: str = "midpoint",
    ) -> Tensor:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        cc = ds_t.f.model_in.where(am, ds_t.f.in_cc)

        if method == "rk4":
            def vm(*args, **kwargs):
                return self.model(*args, **kwargs).clone()
        else:
            vm = self.model
        solver = ODESolver(velocity_model=vm)

        self.model.no_detach = True
        try:
            _, log_ll = solver.compute_likelihood(
                x_1=cc, log_p0=log_p0,
                mask=ds_t.m.full, attn_mask=ds_t.am.full,
                pdgids=pdgids_idx, types=self.types_embd,
                step_size=step_size, method=method,
            )
        finally:
            self.model.no_detach = False
        return log_ll

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        reverse: bool = False,
        split_size: int | None = None,
        step_size: float = 0.04,
        method: str = "midpoint",
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        cc = ds_t.f.model_in.where(am, ds_t.f.in_cc)
        if x_init is not None and x_init.shape == cc.shape:
            x_init = x_init.where(am, _F(x_init).in_p)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        if time_grid is None:
            if method in ("euler", "midpoint"):
                n = max(round(1.0 / step_size), 1)
                time_grid = torch.linspace(0.0, 1.0, n + 1, device=x_init.device, dtype=x_init.dtype)
            else:
                time_grid = x_init.new_tensor([0.0, 1.0])
            if reverse:
                time_grid = time_grid.flip(0)

        if method == "rk4":
            def vm(*args, **kwargs):
                return self.model(*args, **kwargs).clone()
        else:
            vm = self.model
        solver = ODESolver(velocity_model=vm)

        def _sample(x_init, mask, attn_mask, pdgids_idx):
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd, pdgids=pdgids_idx)
            if method == "midpoint":
                return self._midpoint_steps(
                    x_init, time_grid, return_intermediates=return_intermediates, **extras,
                )
            if method == "euler":
                return self._euler_steps(
                    x_init, time_grid, return_intermediates=return_intermediates, **extras,
                )
            return solver.sample(
                x_init=x_init, time_grid=time_grid,
                step_size=step_size, method=method,
                return_intermediates=return_intermediates, **extras,
            )

        return self.chunked(
            _sample, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx,
            split_size=split_size, cat_dim=-3,
        )

    def _midpoint_steps(
        self, x: Tensor, time_grid: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> Tensor:
        xs  = [x]
        man = self.model.manifold
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt     = t_b - t_a
            v1     = self.model(x, t_a, d=dt / 2, **extras)
            x_half = man.expmap(x, dt / 2 * v1)
            v2     = self.model(x_half, t_a + dt / 2, **extras)
            x      = man.expmap(x, dt * man.proju(x, v2))
            if return_intermediates:
                xs.append(x)
        return torch.stack(xs) if return_intermediates else x

    def _euler_steps(
        self, x: Tensor, time_grid: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> Tensor:
        xs  = [x]
        gen = (extras["mask"] == 1).unsqueeze(-1)
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt = t_b - t_a
            s  = self.model(x, t_a, d=dt, **extras)
            x  = torch.where(gen, self.model.manifold.expmap(x, dt * s), x)
            if return_intermediates:
                xs.append(x)
        return torch.stack(xs) if return_intermediates else x
