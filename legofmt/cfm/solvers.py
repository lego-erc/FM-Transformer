"""ODE solving / sampling for the Riemannian FM model, and the ``forward`` that drives it from ``odeint_conf``.

Mixin methods rather than a standalone solver object: ``lego_eval`` version-probes them with
``hasattr(ltng, "_midpoint_steps")`` / ``hasattr(lego, "log_likelihood")``, so they must stay bound on the
LightningModule. Members resolved through ``self`` and owned by ``LEGOLtng``: ``model``, ``types_embd``, ``rc``,
``gen_base_wrapper``, ``convert_pdgids``, ``_opt_eval``.
"""

import contextlib
from functools import partialmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from flow_matching.solver import ODESolver

from legofmt.data.struct import DataStruct, _F

from legofmt.cfm.project_model import is_compiled, uncompiled
from legofmt.compat import dynamic_shape_attention

from legofmt.mod_comps.config import _amp_dtype, resolve_len_buckets

_STEPPED = ("euler", "midpoint", "rk4")   # hand-unrolled manifold solvers (Solvers._steps); others go to ODESolver


class Solvers:
    """Sampling / likelihood mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None, buckets=None):
        """Apply ``fn`` in ``split_size`` chunks along ``dim`` and concatenate along
        ``cat_dim`` (``-3`` is the batch axis of both a ``(B, L, C)`` solution and a
        ``(T, B, L, C)`` intermediates stack). With ``buckets`` every chunk is padded
        to the smallest bucket that holds it, so a compiled ``fn`` only ever sees
        ``len(buckets)`` batch shapes."""
        cat_dim = dim if cat_dim is None else cat_dim
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
        if self.rc.n_lat:
            raise NotImplementedError("log_likelihood does not cover the model_conf.edep_ae latent token.")
        if self.rc.edep_whiten:
            raise NotImplementedError("log_likelihood under model_conf.edep_whiten would miss the whitening's Jacobian")
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        cc = ds_t.f.model_in.where(am, ds_t.f.in_cc)

        vm = (lambda *a, **kw: self.model(*a, **kw).clone()) if method == "rk4" else self.model
        solver = ODESolver(velocity_model=vm)

        self.model.no_detach = True
        try:
            _, log_ll = solver.compute_likelihood(
                x_1=cc, log_p0=log_p0,
                mask=ds_t.m.full, attn_mask=ds_t.am.full,
                pdgids=pdgids_idx, types=self.types_embd[:, :ds_t.am.full.shape[1]],
                step_size=step_size, method=method,
            )
        finally:
            self.model.no_detach = False
        return log_ll

    @torch.no_grad()
    def solve(self, *args, **kwargs) -> Tensor:
        """``_solve`` under ``torch.no_grad``: every sampling caller; ``_solve`` keeps the graph (density design)."""
        return self._solve(*args, **kwargs)

    def _solve(
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
    ) -> Tensor:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)
        if x_init.shape == ds_t.f.model_in.shape:
            x_init = x_init.where(ds_t.am.full.unsqueeze(-1), _F(x_init).in_p)
        if self.rc.edep_whiten:
            x_init = self._edep_whiten(x_init, ds_t)
        # model_conf.edep_ae: the latent z (B, K) starts from N(0, 1) and is integrated with the event; it travels
        # through chunked / bucketed packed into R extra rows of the output and is split off below
        K, C = self.rc.n_lat, x_init.shape[-1]
        R = -(-K // C)
        z_init = (torch.randn(x_init.shape[0], K, device=x_init.device, dtype=x_init.dtype),) if K else ()

        if time_grid is None:  # adaptive ODESolver methods get the bare [0, 1]
            n = max(round(1.0 / step_size), 1) if method in _STEPPED else 1
            time_grid = torch.linspace(0.0, 1.0, n + 1, device=x_init.device, dtype=x_init.dtype)
            if reverse:
                time_grid = time_grid.flip(0)
        else:  # explicit grids (e.g. a YAML list) are used as given; ``reverse`` only flips the built one
            time_grid = torch.as_tensor(time_grid, dtype=x_init.dtype, device=x_init.device)

        def _sample(x_init, mask, attn_mask, pdgids_idx, *z):
            # contiguous: a dynamic graph is guarded on the strides / offsets of the bucket and chunk views
            x_init, mask, attn_mask, pdgids_idx, *z = (a.contiguous() for a in (x_init, mask, attn_mask, pdgids_idx, *z))
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd[:, :mask.shape[1]].contiguous(),
                          pdgids=pdgids_idx)
            if method in _STEPPED:
                out = self._steps(method, x_init, time_grid, return_intermediates, z=z[0] if z else None, **extras)
                if not z:
                    return out
                x, zz = out
                return torch.cat((x, F.pad(zz, (0, R * C - K)).unflatten(-1, (R, C))), dim=-2)
            if z:
                raise NotImplementedError(f"method {method!r} does not integrate the model_conf.edep_ae latent.")
            return ODESolver(velocity_model=self.model).sample(
                x_init=x_init, time_grid=time_grid, step_size=step_size, method=method,
                return_intermediates=return_intermediates, **extras,
            )

        pm = uncompiled(self.model)
        old, pm.no_detach = pm.no_detach, True   # the field sees x with its graph (inert under solve's no_grad)
        try:
            sols = self.chunked(
                _sample, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx, *z_init,
                split_size=split_size, cat_dim=-3, buckets=buckets,
            )
        finally:
            pm.no_detach = old
        if K:
            L = ds_t.am.full.shape[1]
            sols, z = sols[..., :L, :], sols[..., L:, :].flatten(-2)[..., :K]
            self._edep_latents = z if z.dim() == 2 else z[-1]  # the final state's, with return_intermediates
        return self._edep_whiten(sols, ds_t, inverse=True) if self.rc.edep_whiten else sols

    def _field(self, x: Tensor, t, z: Tensor | None, **kw) -> tuple[Tensor, Tensor | None]:
        """The velocity, and that of the ``model_conf.edep_ae`` latent ``z`` (``None`` without it)."""
        return (self.model(x, t, **kw), None) if z is None else self.model(x, t, z=z, **kw)

    def _steps(self, method: str, x: Tensor, time_grid: Tensor, return_intermediates: bool = False,
               z: Tensor | None = None, **extras):
        """Manifold ``euler`` / ``midpoint`` / ``rk4`` steps on the ``mask == 1`` slots, the ``model_conf.edep_ae``
        latent ``z`` integrated alongside. Returns the final state (or the stack of every step's), paired with the
        latent's when there is one."""
        xs, zs = [x], [z]
        man = self.model.manifold
        gen = (extras["mask"] == 1).unsqueeze(-1)
        zz  = lambda dz: None if z is None else z + dz()   # z after an increment, lazily (no latent: None)
        cl  = lambda a: None if a is None else a.clone()
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt = t_b - t_a
            if method == "euler":  # d = dt: the flow map over the whole step
                s, sz = self._field(x, t_a, z, d=dt, **extras)
                x_b, z_b = man.expmap(x, dt * s), zz(lambda: dt * sz)
            elif method == "midpoint":
                # v1 gets d = dt/2, the flow map for the half-step it takes; v2 stays instantaneous because it is
                # applied as a full step from x (tests/test_midpoint_flow_map.py). Its d = 0 is a tensor, not None:
                # a None is a separate dynamo guard, i.e. a second compiled graph.
                v1, w1 = self._field(x, t_a, z, d=dt / 2, **extras)
                v2, w2 = self._field(man.expmap(x, dt / 2 * v1), t_a + dt / 2, zz(lambda: dt / 2 * w1),
                                     d=torch.zeros_like(dt), **extras)
                x_b, z_b = man.expmap(x, dt * man.proju(x, v2)), zz(lambda: dt * w2)
            else:
                # torchdiffeq's rk4 (the 3/8 rule) as a projection method: stages combine in the ambient space, every
                # step ends on the manifold (through the Euclidean ODESolver the sphere factors drifted off it,
                # shortening every later angular step). Instantaneous velocity (d = 0); the clones keep k1..k3 alive
                # across CUDA-graph replays.
                k1, l1 = map(cl, self._field(x, t_a, z, **extras))
                k2, l2 = map(cl, self._field(x + dt * k1 / 3, t_a + dt / 3, zz(lambda: dt * l1 / 3), **extras))
                k3, l3 = map(cl, self._field(x + dt * (k2 - k1 / 3), t_a + dt * 2 / 3,
                                             zz(lambda: dt * (l2 - l1 / 3)), **extras))
                k4, l4 = self._field(x + dt * (k1 - k2 + k3), t_b, zz(lambda: dt * (l1 - l2 + l3)), **extras)
                x_b = man.projx(x + dt * (k1 + 3 * (k2 + k3) + k4) / 8)
                z_b = zz(lambda: dt * (l1 + 3 * (l2 + l3) + l4) / 8)
            x, z = torch.where(gen, x_b, x), z_b
            if return_intermediates:
                xs.append(x)
                zs.append(z)
        out = torch.stack(xs) if return_intermediates else x
        return out if z is None else (out, torch.stack(zs) if return_intermediates else z)

    _midpoint_steps = partialmethod(_steps, "midpoint")  # lego_eval probes hasattr(ltng, "_midpoint_steps")

    def _solve_by_length(self, ds_t: DataStruct, base: Tensor, **kw) -> Tensor:
        L_max = ds_t.am.full.shape[1]
        # odeint_conf may gain len_buckets after construction (lego_eval's per-model odeint override)
        buckets = self.rc.gen_len_buckets or resolve_len_buckets(self.rc.odeint_conf.get("len_buckets"), L_max)
        if buckets is None:
            return self._solve(ds_t, x_init=base, **kw)
        buckets = [min(b, L_max) for b in buckets]
        lengths, order = ds_t.am.full.sum(1).sort()
        counts = torch.bincount(lengths, minlength=L_max + 1).cumsum(0)[buckets].tolist()
        ds_t, base = ds_t[order], base[order]
        sols, lats, a = [], [], 0
        for L, b in zip(buckets, counts):
            if b > a:
                sol = self._solve(ds_t[a:b].trim(L), x_init=base[a:b, :L], **kw)
                sols.append(F.pad(sol, (0, 0, 0, L_max - L)))
                lats.append(self._edep_latents)
            a = b
        if self.rc.n_lat:
            self._edep_latents = torch.cat(lats).index_select(0, order.argsort())
        return torch.cat(sols, dim=-3).index_select(-3, order.argsort())

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        return self._forward(batch, _batch_idx)

    def _forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        """``forward`` without ``torch.no_grad`` (the density design differentiates through it)."""
        self._to_eval()

        cfg = self.rc.odeint_conf
        if cfg.get("fwd_compile", False) and not (is_compiled(self.model) or is_compiled(self.model.vf)):
            dynamic = cfg.get("compile_dynamic", False)
            if dynamic:
                dynamic_shape_attention()
            # CUDA graphs are recorded per distinct shape and kept: a dynamic graph takes them only when compile_buckets
            # bounds the row counts
            self.model = torch.compile(self.model, dynamic=dynamic, mode="reduce-overhead"
                                       if not dynamic or cfg.get("compile_buckets") else None)
            len_b = self.rc.gen_len_buckets or resolve_len_buckets(cfg.get("len_buckets"), self.rc.max_seq_l)
            if len_b is not None and not dynamic:
                torch._dynamo.config.recompile_limit += len(len_b) * max(1, len(cfg.get("compile_buckets") or ()))

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
        if self.sym is not None:
            face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
            g    = torch.randint(8, face.shape, device=face.device) if self.rc.sym_aug or cfg.get("sym_rand") else None
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd, g=g), ds_t.m.full, ds_t.am.full)
        base   = self.gen_base_wrapper(ds_t)
        pdgids = ds_t.f.pdgids
        am     = ds_t.am.full.unsqueeze(-1)

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am, torch.nan)
        else:
            amp = _amp_dtype(cfg["amp"]) if "amp" in cfg else self.rc.amp_dtype
            with contextlib.nullcontext() if amp is None else torch.autocast(base.device.type, dtype=amp):
                sols = self._solve_by_length(
                    ds_t, base,
                    split_size=cfg.get("split_size"),
                    step_size=cfg.get("step_size", 0.04),
                    method=cfg.get("method", "midpoint"),
                    time_grid=cfg.get("time_grid"),
                    return_intermediates=cfg.get("return_timesteps", False),
                    # a dynamic graph still specialises on a 1-row batch: tile those to 2
                    buckets=cfg.get("compile_buckets") or ((2,) if cfg.get("compile_dynamic") else None),
                )
            sols = sols.float().masked_fill_(~am, torch.nan)
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
                keep = torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid)) | (pdgids_idx == 0)
                sols.masked_fill_(~keep, torch.nan)
                pdgids = pdgids.masked_fill(~keep, 0)

        each = lambda f, s: torch.stack([f(x) for x in s]) if s.dim() == 4 else f(s)   # every return_timesteps state
        if self.rc.dir_zoom:
            sols = each(lambda s: self._zoom_dirs(s, ds_t.am.full, fwd, inverse=True, pdgids=ds_t.f.pdgids), sols)
        if self.sym is not None:
            sols = each(lambda s: self._canon_dirs(s, face, fwd, inverse=True, g=g), sols)
        if sols.dim() == 4:
            pdgids = pdgids.unsqueeze(0).expand(sols.shape[0], -1, -1, -1)
        return torch.cat((sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full
