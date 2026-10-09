"""ODE solving / sampling for the Riemannian FM model, and the ``forward``
that drives it from ``odeint_conf``.

Split out of ``main/modules.py``. These are mixin methods rather than a
standalone solver object on purpose: ``lego_eval`` version-probes them with
``hasattr(ltng, "_midpoint_steps")`` / ``hasattr(lego, "log_likelihood")``, so
they must stay bound on the LightningModule.

Members resolved through ``self`` and owned by ``LEGOLtng``: ``model``,
``types_embd``, ``rc``, ``gen_base_wrapper``, ``convert_pdgids``, ``_opt_eval``.
"""

import contextlib

import torch
from torch import Tensor

from flow_matching.solver import ODESolver

from legofmt.data.struct import DataStruct, _F

from legofmt.cfm.project_model import is_compiled

from legofmt.mod_comps.config import _amp_dtype


class Solvers:
    """Sampling / likelihood mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None, buckets=None):
        """Apply ``fn`` in ``split_size`` chunks along ``dim`` and concatenate along
        ``cat_dim`` (``-3`` is the batch axis of both a ``(B, L, C)`` solution and a
        ``(T, B, L, C)`` intermediates stack). With ``buckets`` every chunk is padded
        to the smallest bucket that holds it, so a compiled ``fn`` only ever sees
        ``len(buckets)`` batch shapes."""
        if cat_dim is None:
            cat_dim = dim
        if split_size is None or split_size >= tensors[0].shape[dim]:
            return self._bucketed(fn, tensors, dim, cat_dim, buckets)
        out = [self._bucketed(fn, chunk, dim, cat_dim, buckets)
               for chunk in zip(*(t.split(split_size, dim) for t in tensors))]
        return torch.cat(out, dim=cat_dim)

    @staticmethod
    def _bucketed(fn, tensors, dim, cat_dim, buckets):
        n = tensors[0].shape[dim]
        target = next((b for b in sorted(buckets or ()) if b >= n), None)
        if target is None or target == n:
            return fn(*tensors)
        # tile the chunk's own rows: valid inputs, discarded after the solve
        reps = -(-target // n)
        padded = [torch.cat([t] * reps, dim).narrow(dim, 0, target) for t in tensors]
        return fn(*padded).narrow(cat_dim, 0, n)

    def _to_eval(self) -> None:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

    def _prep_solve(
        self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
    ) -> tuple[DataStruct, Tensor]:
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        self._to_eval()
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
        buckets: list[int] | None = None,
        return_species: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)
        if x_init.shape == ds_t.f.model_in.shape:
            x_init = x_init.where(am, _F(x_init).in_p)

        if time_grid is None:
            if method in ("euler", "midpoint"):
                n = max(round(1.0 / step_size), 1)
                time_grid = torch.linspace(0.0, 1.0, n + 1, device=x_init.device, dtype=x_init.dtype)
            else:
                time_grid = x_init.new_tensor([0.0, 1.0])
            if reverse:
                time_grid = time_grid.flip(0)

        sp     = pdgids_idx.reshape(*ds_t.m.full.shape).long()
        sp_gen = self._species_gen(ds_t.m.full, sp)
        sp     = torch.where(
            sp_gen, torch.randint_like(sp, self.rc.model_args["npdgids"]), sp)

        def _sample(x_init, mask, attn_mask, sp, sp_gen):
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd, pdgids=sp)
            if method == "midpoint":
                out, sp_out = self._midpoint_steps(
                    x_init, time_grid, sp_gen=sp_gen,
                    return_intermediates=return_intermediates, **extras,
                )
            elif method == "euler":
                out, sp_out = self._euler_steps(
                    x_init, time_grid, sp_gen=sp_gen,
                    return_intermediates=return_intermediates, **extras,
                )
            else:
                vm = (lambda *a, **k: self.model(*a, **k).clone()) if method == "rk4" else self.model
                out = ODESolver(velocity_model=vm).sample(
                    x_init=x_init, time_grid=time_grid,
                    step_size=step_size, method=method,
                    return_intermediates=return_intermediates, **extras,
                )
                sp_out = sp
            if return_intermediates:  # out is (T, B, L, C); the class is constant
                sp_out = sp_out.unsqueeze(0).expand(out.shape[0], -1, -1)
            return torch.cat((out, sp_out.unsqueeze(-1).to(out.dtype)), dim=-1)

        res = self.chunked(
            _sample, x_init, ds_t.m.full, ds_t.am.full, sp, sp_gen,
            split_size=split_size, cat_dim=-3, buckets=buckets,
        )
        x, sp_out = res[..., :-1], res[..., -1].long()
        if return_intermediates:  # the class is constant across the stack
            sp_out = sp_out[-1]
        return (x, sp_out) if return_species else x

    def _species_gen(self, mask: Tensor, sp: Tensor) -> Tensor:
        """Which slots generate their own species: transported outgoing slots that
        arrive unassigned. A pre-filled class pins the slot, which is how a caller
        conditions on a known multiplicity."""
        return (mask == 1) & (sp == 0) & (
            torch.arange(mask.shape[1], device=mask.device) >= self.rc.n_prefix + 1)

    def _species_jump(self, sp: Tensor, logits: Tensor, sp_gen: Tensor,
                      t_a: Tensor, dt: Tensor) -> Tensor:
        """One mixture-path CTMC step: resample the class from the predicted
        posterior over ``x_1`` at rate ``dt / (1 - t)``. Gumbel-argmax draws that
        categorical without materialising a softmax. A reverse grid gives a negative
        rate, which clamps to zero -- reverse solves do not regenerate."""
        lg     = logits.float()
        jump   = torch.rand_like(lg[..., 0]) < (dt / (1 - t_a)).clamp(0.0, 1.0)
        gumbel = -torch.rand_like(lg).log().neg().log()
        return torch.where(jump & sp_gen, (lg + gumbel).argmax(-1), sp)

    def _midpoint_steps(
        self, x: Tensor, time_grid: Tensor, sp_gen: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> tuple[Tensor, Tensor]:
        """Manifold midpoint steps on the ``mask == 1`` slots. ``v1`` is queried with
        ``d = dt/2`` (the flow map for the half-step it takes); ``v2`` stays
        instantaneous because it is applied as a full step from ``x``
        (tests/test_midpoint_flow_map.py)."""
        xs  = [x]
        man = self.model.manifold
        gen = (extras["mask"] == 1).unsqueeze(-1)
        lg = None
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt     = t_b - t_a
            v1     = self.model(x, t_a, d=dt / 2, **extras)
            x_half = man.expmap(x, dt / 2 * v1)
            v2, lg = self.model(x_half, t_a + dt / 2, return_species=True, **extras)
            x      = torch.where(gen, man.expmap(x, dt * man.proju(x, v2)), x)
            extras["pdgids"] = self._species_jump(extras["pdgids"], lg, sp_gen, t_a, dt)
            if return_intermediates:
                xs.append(x)
        if lg is not None:  # settle on the final posterior, not a late resample
            extras["pdgids"] = torch.where(sp_gen, lg.argmax(-1), extras["pdgids"])
        return (torch.stack(xs) if return_intermediates else x), extras["pdgids"]

    def _euler_steps(
        self, x: Tensor, time_grid: Tensor, sp_gen: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> tuple[Tensor, Tensor]:
        """Manifold Euler steps with ``d = dt`` handed to the flow map; only the
        ``mask == 1`` slots are transported."""
        xs  = [x]
        gen = (extras["mask"] == 1).unsqueeze(-1)
        lg = None
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt     = t_b - t_a
            s, lg  = self.model(x, t_a, d=dt, return_species=True, **extras)
            x      = torch.where(gen, self.model.manifold.expmap(x, dt * s), x)
            extras["pdgids"] = self._species_jump(extras["pdgids"], lg, sp_gen, t_a, dt)
            if return_intermediates:
                xs.append(x)
        if lg is not None:  # settle on the final posterior, not a late resample
            extras["pdgids"] = torch.where(sp_gen, lg.argmax(-1), extras["pdgids"])
        return (torch.stack(xs) if return_intermediates else x), extras["pdgids"]

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        self._to_eval()

        cfg = self.rc.odeint_conf
        if cfg.get("fwd_compile", False) and not (
            is_compiled(self.model) or is_compiled(self.model.vf)
        ):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        if self.sym is not None:
            fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
            face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
            g    = (torch.randint(8, face.shape, device=face.device)
                    if (self.rc.sym_aug or cfg.get("sym_rand", False)) else None)
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd, g=g), ds_t.m.full, ds_t.am.full)
        base = self.gen_base_wrapper(ds_t)

        n0     = self.rc.n_prefix + 1
        raw    = ds_t.f.pdgids
        sp     = (raw.int() if self.rc.pdgid_is_idx else self.convert_pdgids(raw)
                  ).reshape(*ds_t.m.full.shape).long()
        am_out = ds_t.am.full

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am_out.unsqueeze(-1), torch.nan)
        else:
            amp = (_amp_dtype(cfg["amp"]) if "amp" in cfg
                   else self.rc.amp_dtype)
            with (contextlib.nullcontext() if amp is None
                  else torch.autocast(base.device.type, dtype=amp)):
                sols, sp = self.solve(
                    ds_t, x_init=base,
                    split_size=cfg.get("split_size"),
                    step_size=cfg.get("step_size", 0.04),
                    method=cfg.get("method", "midpoint"),
                    time_grid=cfg.get("time_grid"),
                    return_intermediates=cfg.get("return_timesteps", False),
                    buckets=cfg.get("compile_buckets"),
                    return_species=True,
                )
            sols = sols.float()
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                sp = sp.masked_fill(
                    ~(torch.isin(sp, self.convert_pdgids(filter_pdgid)) | (sp == 0)), 0)
            am_out = ds_t.am.full.clone()
            am_out[:, n0:] = torch.where(
                ds_t.m.full[:, n0:] == 1, sp[:, n0:] > 0, am_out[:, n0:])
            sols = sols.masked_fill_(~am_out.unsqueeze(-1), torch.nan)

        pdgids = sp.unsqueeze(-1)
        if not self.rc.pdgid_is_idx:  # back to raw ids, pad class -> 0
            tmpl   = self.pdgids_template
            pdgids = torch.cat((tmpl.new_zeros(1), tmpl))[sp].unsqueeze(-1)
        pdgids = pdgids.to(sols.dtype)

        if self.sym is not None:
            if sols.dim() == 4:
                sols = torch.stack([self._canon_dirs(s, face, fwd, inverse=True, g=g) for s in sols])
            else:
                sols = self._canon_dirs(sols, face, fwd, inverse=True, g=g)

        if sols.dim() == 4:
            pdgids = pdgids.unsqueeze(0).expand(sols.shape[0], -1, -1, -1)
        return torch.cat((sols, pdgids), dim=-1), ds_t.m.full, am_out
