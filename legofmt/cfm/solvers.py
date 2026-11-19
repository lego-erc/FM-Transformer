"""Integrating the flow: the manifold solvers, ``solve``, and the ``forward`` that generates a batch.

One step from ``t_a`` to ``t_b = t_a + dt``. ``v(x, t; d)`` is the field; ``d`` is the step size a flow-map field
(``step_cond``) is told it takes, and ``d = 0`` is the instantaneous velocity.

```
euler      x_b = exp_x(dt v(x, t_a; dt))

midpoint   x_m = exp_x(dt/2 v(x, t_a; dt/2))
           x_b = exp_x(dt proj_x v(x_m, t_a + dt/2; 0))

rk4        k_1 = v(x, t_a; 0)
           k_2 = v(x + dt k_1 / 3,             t_a + dt/3;   0)
           k_3 = v(x + dt (k_2 - k_1 / 3),     t_a + 2 dt/3; 0)
           k_4 = v(x + dt (k_1 - k_2 + k_3),   t_b;          0)
           x_b = proj_M(x + dt (k_1 + 3 (k_2 + k_3) + k_4) / 8)
```

``exp``, ``proj_x`` (onto the tangent space at ``x``) and ``proj_M`` (onto the manifold) are the product manifold's.
Only generated tokens move. The ``model_conf.edep_ae`` latent ``z`` is Euclidean: it takes the same steps without
the maps.

``Solvers`` is a mixin rather than a solver object: ``lego_eval`` version-probes it with
``hasattr(ltng, "_midpoint_steps")`` / ``hasattr(lego, "log_likelihood")``, so the methods must stay bound on the
LightningModule. It reads from ``LEGOLtng``: ``model``, ``types_embd``, ``rc``, ``gen_base_wrapper``,
``_edep_whiten``, ``convert_pdgids``, ``sym``, ``_canon_dirs``, ``_zoom_dirs``, ``_opt_eval``.

Any other ``method`` (flow_matching's ``ODESolver``, e.g. ``dopri5``) is :mod:`legofmt.legacy.solvers`.
"""

import contextlib
from functools import partialmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from flow_matching.solver import ODESolver

from legofmt.rng import LATENT, SYM_FLOW
from legofmt.data.struct import DataStruct, _F

from legofmt.cfm.project_model import is_compiled, uncompiled
from legofmt.compat import dynamic_shape_attention
from legofmt.legacy.solvers import odesolver_sample

from legofmt.mod_comps.config import _amp_dtype, resolve_len_buckets


# ---- one step ----------------------------------------------------------------------------------------------------

def _euler(field, man, x, z, t_a, t_b):
    """Takes one Euler step from ``t_a`` to ``t_b``.

    The field is told the step it takes, ``d = dt``.

    Args:
        field: ``field(x, z, t, d=...)``, the velocities ``(v, w)`` of the state and the latent.
        man:   the product manifold.
        x:     (B, L, 7), the state at ``t_a``.
        z:     (B, K), the latent at ``t_a``; K = 0: no latent.
        t_a:   scalar, the step's start.
        t_b:   scalar, the step's end.

    Returns:
        ``(x_b, z_b)``, the state and the latent at ``t_b``.
    """
    dt   = t_b - t_a
    v, w = field(x, z, t_a, d=dt)
    return man.expmap(x, dt * v), z + dt * w


def _midpoint(field, man, x, z, t_a, t_b):
    """Takes one midpoint step from ``t_a`` to ``t_b``.

    ``v1`` takes the half-step, so its ``d`` is ``dt / 2``; ``v2`` is applied as a full step from ``x`` and stays
    instantaneous (tests/test_midpoint_flow_map.py). Its ``d = 0`` is a tensor, not None: a None is a separate
    dynamo guard, i.e. a second compiled graph.

    Args:
        field: ``field(x, z, t, d=...)``, the velocities ``(v, w)`` of the state and the latent.
        man:   the product manifold.
        x:     (B, L, 7), the state at ``t_a``.
        z:     (B, K), the latent at ``t_a``; K = 0: no latent.
        t_a:   scalar, the step's start.
        t_b:   scalar, the step's end.

    Returns:
        ``(x_b, z_b)``, the state and the latent at ``t_b``.
    """
    dt     = t_b - t_a
    h      = dt / 2
    v1, w1 = field(x, z, t_a, d=h)
    v2, w2 = field(man.expmap(x, h * v1), z + h * w1, t_a + h, d=torch.zeros_like(dt))
    return man.expmap(x, dt * man.proju(x, v2)), z + dt * w2


def _rk4(field, man, x, z, t_a, t_b):
    """Takes one fourth-order Runge-Kutta step from ``t_a`` to ``t_b``.

    torchdiffeq's rk4 (the 3/8 rule) as a projection method: the stages combine in the ambient space, every step
    ends on the manifold (through the Euclidean ODESolver the sphere factors drifted off it, shortening every later
    angular step). All stages are instantaneous velocities (no ``d``). The clones keep ``k1..k3`` alive across
    CUDA-graph replays.

    Args:
        field: ``field(x, z, t, d=...)``, the velocities ``(v, w)`` of the state and the latent.
        man:   the product manifold.
        x:     (B, L, 7), the state at ``t_a``.
        z:     (B, K), the latent at ``t_a``; K = 0: no latent.
        t_a:   scalar, the step's start.
        t_b:   scalar, the step's end.

    Returns:
        ``(x_b, z_b)``, the state and the latent at ``t_b``.
    """
    dt     = t_b - t_a
    k1, l1 = (a.clone() for a in field(x, z, t_a))
    k2, l2 = (a.clone() for a in field(x + dt * k1 / 3, z + dt * l1 / 3, t_a + dt / 3))
    k3, l3 = (a.clone() for a in field(x + dt * (k2 - k1 / 3), z + dt * (l2 - l1 / 3), t_a + dt * 2 / 3))
    k4, l4 = field(x + dt * (k1 - k2 + k3), z + dt * (l1 - l2 + l3), t_b)
    return man.projx(x + dt * (k1 + 3 * (k2 + k3) + k4) / 8), z + dt * (l1 + 3 * (l2 + l3) + l4) / 8


_STEPPED = {"euler": _euler, "midpoint": _midpoint, "rk4": _rk4}


def _time_grid(time_grid, step_size: float, method: str, reverse: bool, like: Tensor) -> Tensor:
    """Builds the solve's time grid: ``round(1 / step_size)`` equal steps over [0, 1], flipped for ``reverse``.

    An explicit ``time_grid`` (e.g. a YAML list) is used as given; ``reverse`` only flips the built one.
    """
    if time_grid is not None:
        return torch.as_tensor(time_grid, dtype=like.dtype, device=like.device)
    n    = max(round(1.0 / step_size), 1) if method in _STEPPED else 1   # an ODESolver method: the bare [0, 1]
    grid = torch.linspace(0.0, 1.0, n + 1, device=like.device, dtype=like.dtype)
    return grid.flip(0) if reverse else grid


def _pack_latent(x: Tensor, z: Tensor) -> Tensor:
    """Appends the latent ``z`` (..., B, K) to the state ``x`` (..., B, L, C) as ``ceil(K / C)`` extra tokens.

    One tensor then travels through ``chunked``; ``_split_latent`` undoes it.
    """
    K, C = z.shape[-1], x.shape[-1]
    R    = -(-K // C)
    return torch.cat((x, F.pad(z, (0, R * C - K)).unflatten(-1, (R, C))), dim=-2)


def _split_latent(sols: Tensor, L: int, K: int) -> tuple[Tensor, Tensor]:
    """Splits what ``_pack_latent`` joined into the state (..., B, L, C) and the latent (..., B, K)."""
    return sols[..., :L, :], sols[..., L:, :].flatten(-2)[..., :K]


class Solvers:
    """Sampling / likelihood mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    # ---- generating a batch -------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None, rng=None) -> tuple:
        return self._forward(batch, _batch_idx, rng=rng)

    def _forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None, rng=None) -> tuple:
        """Generates a batch: ``forward`` without ``torch.no_grad`` (the density design differentiates through it).

        Samples the base, solves as ``odeint_conf`` says, and maps the directions back from the frame the flow works
        in (``canon_sym``, ``dir_zoom``).

        Args:
            batch:      the conditions, a ``DataStruct`` or the three tensors it is built from.
            _batch_idx: unused.
            rng:        keys every random draw on its event (``legofmt.rng.EventRNG``, one key per row of ``batch``).

        Returns:
            ``(sols, mask, attn_mask)``: the samples (B, L, 8), pdgids last, and the batch's two masks.
        """
        self._to_eval()
        cfg = self.rc.odeint_conf
        self._compile_once(cfg)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
        if self.sym is not None:   # canon_sym: every entry face onto +x; sym_aug / sym_rand: a random element g
            face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
            g    = ((torch.randint(8, face.shape, device=face.device) if rng is None
                     else rng.randint(8, SYM_FLOW).view(face.shape).to(face.device))
                    if self.rc.sym_aug or cfg.get("sym_rand") else None)
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd, g=g), ds_t.m.full, ds_t.am.full)
        base   = self.gen_base_wrapper(ds_t, rng=rng)
        pdgids = ds_t.f.pdgids
        am     = ds_t.am.full.unsqueeze(-1)

        if cfg.get("return_base", False):
            sols = self._edep_whiten(base, ds_t, inverse=True).masked_fill(~am, torch.nan)
        else:
            amp = _amp_dtype(cfg["amp"]) if "amp" in cfg else self.rc.amp_dtype
            with contextlib.nullcontext() if amp is None else torch.autocast(base.device.type, dtype=amp):
                sols = self._solve_by_length(
                    ds_t, base, rng=rng,
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

    def _compile_once(self, cfg: dict) -> None:
        """Compiles the field for generation on the first call (``odeint_conf.fwd_compile``)."""
        if not cfg.get("fwd_compile", False) or is_compiled(self.model) or is_compiled(self.model.vf):
            return
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

    def _solve_by_length(self, ds_t: DataStruct, base: Tensor, rng=None, **kw) -> Tensor:
        """Solves the events in groups of similar length (``len_buckets``), each trimmed to its bucket.

        Without buckets it is one ``_solve``. ``base`` is ``x_init``; ``kw`` goes to ``_solve``. Returns the
        solutions in the batch's order, padded back to its length.
        """
        L_max = ds_t.am.full.shape[1]
        # odeint_conf may gain len_buckets after construction (lego_eval's per-model odeint override)
        buckets = self.rc.gen_len_buckets or resolve_len_buckets(self.rc.odeint_conf.get("len_buckets"), L_max)
        if buckets is None:
            return self._solve(ds_t, x_init=base, rng=rng, **kw)
        buckets = [min(b, L_max) for b in buckets]
        lengths, order = ds_t.am.full.sum(1).sort()
        counts = torch.bincount(lengths, minlength=L_max + 1).cumsum(0)[buckets].tolist()
        ds_t, base = ds_t[order], base[order]
        sols, lats, a = [], [], 0
        for L, b in zip(buckets, counts):
            if b > a:
                sol = self._solve(ds_t[a:b].trim(L), x_init=base[a:b, :L],
                                  rng=None if rng is None else rng.sub(order[a:b]), **kw)
                sols.append(F.pad(sol, (0, 0, 0, L_max - L)))
                lats.append(self._edep_latents)
            a = b
        if self.rc.n_lat:
            self._edep_latents = torch.cat(lats).index_select(0, order.argsort())
        return torch.cat(sols, dim=-3).index_select(-3, order.argsort())

    # ---- solving ------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def solve(self, *args, **kwargs) -> Tensor:
        """Runs ``_solve`` under ``torch.no_grad``: every sampling caller.

        ``_solve`` keeps the graph (density design).
        """
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
        rng=None,
    ) -> Tensor:
        """Integrates the flow along a time grid.

        Args:
            ds_t:                 the batch, a ``DataStruct`` or the three tensors it is built from.
            x_init:               (B, L, 7), the start; None: a sample of the base. At ``t = 0`` its E_dep token is in
                the flow's unit, past it a deposit (a state as ``solve`` returns it; ``reverse``: the data).
            reverse:              integrate from 1 to 0.
            split_size:           solve in chunks of that many events.
            step_size:            the step; the grid has ``round(1 / step_size)`` of them.
            method:               ``euler``, ``midpoint`` or ``rk4``.
            time_grid:            an explicit grid, used as given.
            return_intermediates: return the state at every grid point.
            buckets:              pad every chunk to one of these batch sizes (``chunked``).
            rng:                  keys the base and the latent on the event (``legofmt.rng.EventRNG``).

        Returns:
            The state at the grid's end (B, L, 7), or at every grid point (T, B, L, 7); E_dep is a deposit. With
            ``model_conf.edep_ae`` the final latent is left in ``self._edep_latents`` (B, K).
        """
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t, rng=rng)
        if x_init.shape == ds_t.f.model_in.shape:
            x_init = x_init.where(ds_t.am.full.unsqueeze(-1), _F(x_init).in_p)
        K = self.rc.n_lat   # model_conf.edep_ae: the latent z (B, K) starts from N(0, 1), integrated with the event
        z_init = ()
        if K:
            z_init = (torch.randn(x_init.shape[0], K, device=x_init.device, dtype=x_init.dtype) if rng is None
                      else rng.normal(LATENT, torch.arange(K)).to(x_init),)
        time_grid = _time_grid(time_grid, step_size, method, reverse, x_init)
        if time_grid[0] > 0:   # a start past t = 0 is a state as solve returns it (reverse: the data), E_dep a deposit
            x_init = self._edep_whiten(x_init, ds_t)

        def sample(x, mask, attn_mask, pdgids_idx, *z):
            # contiguous: a dynamic graph is guarded on the strides / offsets of the bucket and chunk views
            x, mask, attn_mask, pdgids_idx, *z = (a.contiguous() for a in (x, mask, attn_mask, pdgids_idx, *z))
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd[:, :mask.shape[1]].contiguous(),
                          pdgids=pdgids_idx)
            if method not in _STEPPED: return odesolver_sample(self.model, x, time_grid, step_size, method,
                                                               return_intermediates, z, **extras)
            out = self._steps(method, x, time_grid, return_intermediates, *z, **extras)
            return _pack_latent(*out) if z else out

        pm = uncompiled(self.model)
        old, pm.no_detach = pm.no_detach, True   # the field sees x with its graph (inert under solve's no_grad)
        try:
            sols = self.chunked(
                sample, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx, *z_init,
                split_size=split_size, cat_dim=-3, buckets=buckets,
            )
        finally:
            pm.no_detach = old
        if K:
            sols, z = _split_latent(sols, ds_t.am.full.shape[1], K)
            self._edep_latents = z if z.dim() == 2 else z[-1]  # the final state's, with return_intermediates
        return self._edep_whiten(sols, ds_t, inverse=True)

    def _steps(self, method: str, x: Tensor, time_grid: Tensor, return_intermediates: bool = False,
               z: Tensor | None = None, **extras):
        """Integrates ``x`` along ``time_grid`` with manifold ``euler`` / ``midpoint`` / ``rk4`` steps.

        Only the ``mask == 1`` tokens move; the ``model_conf.edep_ae`` latent ``z`` is integrated alongside.

        Args:
            method:               the step.
            x:                    (B, L, 7), the start.
            time_grid:            the grid.
            return_intermediates: return every grid point's state.
            z:                    (B, K), the latent's start; None: no latent.
            extras:               the field's keyword inputs.

        Returns:
            The final state, or the stack of every grid point's; with ``z`` paired with the latent's.
        """
        step    = _STEPPED[method]
        man     = self.model.manifold
        gen     = (extras["mask"] == 1).unsqueeze(-1)
        field   = lambda x, z, t, **d: self._field(x, t, z, **d, **extras)
        latent  = z is not None
        z       = z if latent else x.new_zeros(len(x), 0)   # no latent: an empty one, which the steps carry along
        xs, zs  = [x], [z]
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            x_b, z = step(field, man, x, z, t_a, t_b)
            x      = torch.where(gen, x_b, x)
            if return_intermediates:
                xs.append(x)
                zs.append(z)
        out = torch.stack(xs) if return_intermediates else x
        return (out, torch.stack(zs) if return_intermediates else z) if latent else out

    _midpoint_steps = partialmethod(_steps, "midpoint")  # lego_eval probes hasattr(ltng, "_midpoint_steps")

    def _field(self, x: Tensor, t, z: Tensor, **kw) -> tuple[Tensor, Tensor]:
        """Evaluates the velocity of the state and of the latent ``z``; an empty ``z`` (no latent) passes through."""
        return (self.model(x, t, **kw), z) if z.shape[-1] == 0 else self.model(x, t, z=z, **kw)

    # ---- likelihood ---------------------------------------------------------------------------------------------

    @torch.no_grad()
    def log_likelihood(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        log_p0,
        step_size: float = 0.04,
        method: str = "midpoint",
    ) -> Tensor:
        """Computes the log-likelihood of the batch's data: the flow integrated back to the base, with its divergence.

        flow_matching's ``ODESolver.compute_likelihood``.

        Args:
            ds_t:      the batch, a ``DataStruct`` or the three tensors it is built from.
            log_p0:    the base's log density, a function of the state at ``t = 0``.
            step_size: the solver's step.
            method:    a torchdiffeq method.

        Returns:
            (B,), the log-likelihood per event.

        Raises:
            NotImplementedError: the model has a latent token (``model_conf.edep_ae``) or a whitened E_dep token
                (``model_conf.edep_whiten``), whose Jacobian this would miss.
        """
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

    # ---- plumbing -----------------------------------------------------------------------------------------------

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

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None, buckets=None):
        """Applies ``fn`` in chunks along ``dim`` and concatenates the results.

        Args:
            fn:         called on each chunk of ``tensors``.
            tensors:    split alike along ``dim``.
            split_size: the chunk size; None: one chunk.
            dim:        the axis to split.
            cat_dim:    the axis to concatenate along; None: ``dim``. ``-3`` is the batch axis of both a ``(B, L, C)``
                solution and a ``(T, B, L, C)`` intermediates stack.
            buckets:    pad every chunk to the smallest bucket that holds it, so a compiled ``fn`` only ever sees
                ``len(buckets)`` batch shapes.

        Returns:
            The concatenated results.
        """
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
