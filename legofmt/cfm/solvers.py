"""Integrating the flow: the manifold solvers, ``solve``, and the ``forward`` that generates a batch.

One step from ``t_a`` to ``t_b = t_a + dt`` of the state ``s = (x, z)`` (``state.py``): the tokens ``x`` and the
``model_conf.edep_ae`` latent ``z``, Euclidean (empty without one). ``v(s, t; d)`` is the field; ``d`` is the step
size a flow-map field (``step_cond``) is told it takes, and ``d = 0`` is the instantaneous velocity.

```
euler      s_b = exp_s(dt v(s, t_a; dt))

midpoint   s_m = exp_s(h v(s, t_a; h)),   h = dt / 2
           s_b = exp_s(dt proj_s v(s_m, t_a + h; 0))

rk4        k1  = v(s, t_a; 0)
           k2  = v(s + dt k1 / 3,           t_a + dt / 3;     0)
           k3  = v(s + dt (k2 - k1 / 3),    t_a + dt 2 / 3;   0)
           k4  = v(s + dt (k1 - k2 + k3),   t_b;              0)
           s_b = proj_M(s + dt (k1 + 3 (k2 + k3) + k4) / 8)
```

``exp``, ``proj_s`` (onto the tangent space at ``s``) and ``proj_M`` (onto the manifold) are the product manifold's,
acting on ``x``. Every step is ``step(field, man, s, t_a, t_b) -> s_b``: ``field(s, t, d=...)`` the velocity at ``s``
(a ``State``), ``man`` the manifold's maps on a state (``LatentManifold``). Only generated tokens move.

``Solvers`` is a mixin rather than a solver object: ``lego_eval`` version-probes it with
``hasattr(ltng, "_midpoint_steps")`` / ``hasattr(lego, "log_likelihood")``, so the methods must stay bound on the
LightningModule. It reads from ``LEGOLtng``: ``model``, ``types_embd``, ``rc``, ``gen_base_wrapper``,
``_edep_whiten``, ``convert_pdgids``, ``_enter_frame`` / ``_leave_frame``, ``_opt_eval``.

Any other ``method`` (flow_matching's ``ODESolver``, e.g. ``dopri5``) is :mod:`legofmt.legacy.solvers`.
"""

import contextlib
from functools import partialmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from flow_matching.solver import ODESolver

from legofmt.rng import LATENT, normal_like, rng_sub
from legofmt.data.struct import DataStruct, _F

from legofmt.cfm.project_model import is_compiled, uncompiled
from legofmt.cfm.state import LatentManifold, State
from legofmt.compat import dynamic_shape_attention
from legofmt.legacy.solvers import odesolver_sample

from legofmt.mod_comps.config import _amp_dtype, resolve_len_buckets


# ---- one step ----------------------------------------------------------------------------------------------------

def _euler(field, man, s, t_a, t_b):
    """Takes one Euler step; the field is told the step it takes, ``d = dt``."""
    dt = t_b - t_a
    return man.expmap(s, dt * field(s, t_a, d=dt))


def _midpoint(field, man, s, t_a, t_b):
    """Takes one midpoint step.

    The half step is told its ``d = h``; the midpoint velocity is applied as a full step from ``s`` and stays
    instantaneous (tests/test_midpoint_flow_map.py). Its ``d = 0`` is a tensor, not None: a None is a separate dynamo
    guard, i.e. a second compiled graph.
    """
    dt  = t_b - t_a
    h   = dt / 2
    s_m = man.expmap(s, h * field(s, t_a, d=h))
    return man.expmap(s, dt * man.proju(s, field(s_m, t_a + h, d=torch.zeros_like(dt))))


def _rk4(field, man, s, t_a, t_b):
    """Takes one fourth-order Runge-Kutta step.

    torchdiffeq's rk4 (the 3/8 rule) as a projection method: the stages combine in the ambient space, every step
    ends on the manifold (through the Euclidean ODESolver the sphere factors drifted off it, shortening every later
    angular step). All stages are instantaneous velocities (no ``d``).
    """
    dt = t_b - t_a
    k1 = field(s, t_a)
    k2 = field(s + dt * k1 / 3,         t_a + dt / 3)
    k3 = field(s + dt * (k2 - k1 / 3),  t_a + dt * 2 / 3)
    k4 = field(s + dt * (k1 - k2 + k3), t_b)
    return man.projx(s + dt * (k1 + 3 * (k2 + k3) + k4) / 8)


_STEPPED = {"euler": _euler, "midpoint": _midpoint, "rk4": _rk4}


class Solvers:
    """Sampling / likelihood mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    # ---- generating a batch -------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None, rng=None) -> tuple:
        """Runs ``_forward`` under ``torch.no_grad``: generates a batch."""
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
        cfg  = self._ready_to_generate()
        ds   = DataStruct(*batch) if isinstance(batch, tuple) else batch
        ds_t, frame = self._enter_frame(ds, aug=self.rc.sym_aug or cfg.get("sym_rand"), rng=rng)   # canon_sym
        x0   = self.gen_base_wrapper(ds_t, rng=rng)
        x1, pdgids = self._solve_as_configured(ds_t, x0, cfg, rng)
        x1   = _on_every_state(lambda x: self._leave_frame(x, ds_t, frame), x1)   # dir_zoom, canon_sym undone
        return _with_pdgids(x1, pdgids), ds_t.m.full, ds_t.am.full

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
        x0   = self._x_start(ds_t, x_init, rng)                  # a sample of the base, or the state given
        z0   = self._z_start(x0, rng)                            # model_conf.edep_ae: z_0 ~ N(0, 1)
        grid = _time_grid(time_grid, step_size, method, reverse, x0)
        if grid[0] > 0:                                          # a start past t = 0 is a deposit: to the flow's unit
            x0 = self._edep_whiten(x0, ds_t)
        x1 = self._solve_in_chunks(ds_t, pdgids_idx, x0, z0, grid, step_size, method, return_intermediates,
                                   split_size, buckets)
        return self._edep_whiten(x1, ds_t, inverse=True)

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
        step  = _STEPPED[method]
        man   = LatentManifold(self.model.manifold)
        field = self._state_field(extras, clone=method == "rk4")
        gen   = (extras["mask"] == 1).unsqueeze(-1)
        s     = State(x, _no_latent(x) if z is None else z)
        path  = [s]
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            s_b = step(field, man, s, t_a, t_b)
            s   = State(torch.where(gen, s_b.x, s.x), s_b.z)    # only the generated tokens move
            if return_intermediates:
                path.append(s)
        return _as_returned(s, path, return_intermediates, latent=z is not None)

    _midpoint_steps = partialmethod(_steps, "midpoint")  # lego_eval probes hasattr(ltng, "_midpoint_steps")

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
        self._refuse_unsupported_likelihood()
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        x1 = torch.where(ds_t.am.full.unsqueeze(-1), ds_t.f.model_in, ds_t.f.in_cc)   # data; padding: the incoming one
        with _field_sees_graph(self.model, after=False):
            _, log_ll = _ode_solver(self.model, method).compute_likelihood(
                x_1=x1, log_p0=log_p0,
                mask=ds_t.m.full, attn_mask=ds_t.am.full,
                pdgids=pdgids_idx, types=self.types_embd[:, :ds_t.am.full.shape[1]],
                step_size=step_size, method=method,
            )
        return log_ll

    # ---- plumbing -----------------------------------------------------------------------------------------------

    def _to_eval(self) -> None:
        """Puts the field in eval mode, and a schedule-free optimizer with it (legacy), once."""
        if self.model.training:
            self.model.eval()
            self._opt_eval()

    def _prep_solve(
        self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
    ) -> tuple[DataStruct, Tensor]:
        """Returns the batch as a ``DataStruct`` and its tokens' species indices; puts the field in eval mode."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        self._to_eval()
        pdgids     = ds_t.f.pdgids
        pdgids_idx = pdgids.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
        return ds_t, pdgids_idx

    def _ready_to_generate(self) -> dict:
        """Puts the model in eval mode and compiles it once; returns ``odeint_conf``."""
        self._to_eval()
        cfg = self.rc.odeint_conf
        self._compile_once(cfg)
        return cfg

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

    def _solve_as_configured(self, ds_t: DataStruct, base: Tensor, cfg: dict, rng) -> tuple[Tensor, Tensor]:
        """Solves from ``base`` as ``odeint_conf`` says; returns the samples, NaN on padding, and their pdgids.

        ``return_base``: the base itself. ``amp``: the autocast dtype. ``filter_pdgid``: every other species NaN, with
        pdgid 0.
        """
        pdgids = ds_t.f.pdgids
        am     = ds_t.am.full.unsqueeze(-1)
        if cfg.get("return_base", False):
            return torch.where(am, self._edep_whiten(base, ds_t, inverse=True), torch.nan), pdgids
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
            pdgids_idx = pdgids.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
            keep = torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid)) | (pdgids_idx == 0)
            sols.masked_fill_(~keep, torch.nan)
            pdgids = torch.where(keep, pdgids, 0)
        return sols, pdgids

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
                                  rng=rng_sub(rng, order[a:b]), **kw)
                sols.append(F.pad(sol, (0, 0, 0, L_max - L)))
                lats.append(self._edep_latents)
            a = b
        if self.rc.n_lat:
            self._edep_latents = torch.cat(lats).index_select(0, order.argsort())
        return torch.cat(sols, dim=-3).index_select(-3, order.argsort())

    def _x_start(self, ds_t: DataStruct, x_init: Tensor | None, rng) -> Tensor:
        """Returns ``x_init``, or a sample of the base; padding holds the incoming particle."""
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t, rng=rng)
        if x_init.shape == ds_t.f.model_in.shape:
            x_init = torch.where(ds_t.am.full.unsqueeze(-1), x_init, _F(x_init).in_p)
        return x_init

    def _z_start(self, x0: Tensor, rng) -> tuple:
        """Samples the latent's start ``(z_0,)``, N(0, 1) (B, K); ``()`` without ``model_conf.edep_ae``."""
        K = self.rc.n_lat
        if not K:
            return ()
        z_like = x0.new_empty(len(x0), K)   # (B, K), in x0's dtype and on its device
        return (normal_like(z_like, rng, LATENT, torch.arange(K)),)

    def _solve_in_chunks(self, ds_t: DataStruct, pdgids_idx: Tensor, x0: Tensor, z0: tuple, grid: Tensor,
                         step_size: float, method: str, return_intermediates: bool, split_size: int | None,
                         buckets: list[int] | None) -> Tensor:
        """Runs ``_steps`` (any other method: the legacy ODESolver) on chunks of the batch (``chunked``).

        The field sees x with its graph. The latent travels as extra tokens (``_pack_latent``); the final one is left
        in ``self._edep_latents``.
        """
        def sample(x, mask, attn_mask, pdgids_idx, *z):
            # contiguous: a dynamic graph is guarded on the strides / offsets of the bucket and chunk views
            x, mask, attn_mask, pdgids_idx, *z = (a.contiguous() for a in (x, mask, attn_mask, pdgids_idx, *z))
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd[:, :mask.shape[1]].contiguous(),
                          pdgids=pdgids_idx)
            if method not in _STEPPED: return odesolver_sample(self.model, x, grid, step_size, method,   # legacy
                                                               return_intermediates, z, **extras)
            out = self._steps(method, x, grid, return_intermediates, *z, **extras)
            return _pack_latent(*out) if z else out

        with _field_sees_graph(uncompiled(self.model)):
            sols = self.chunked(
                sample, x0, ds_t.m.full, ds_t.am.full, pdgids_idx, *z0,
                split_size=split_size, cat_dim=-3, buckets=buckets,
            )
        if z0:
            sols, z = _split_latent(sols, ds_t.am.full.shape[1], self.rc.n_lat)
            self._edep_latents = z if z.dim() == 2 else z[-1]  # the final state's, with return_intermediates
        return sols

    def _state_field(self, extras: dict, clone: bool = False):
        """Returns the field on a ``State``, with the batch's keyword inputs ``extras``.

        An empty latent passes through as its own velocity. ``clone`` keeps the velocities alive across CUDA-graph
        replays (rk4 holds its stages).
        """
        def field(s, t, **d):
            if s.z.shape[-1] == 0:
                v = (self.model(s.x, t, **d, **extras), s.z)
            else:
                v = self.model(s.x, t, z=s.z, **d, **extras)
            return State(*(a.clone() for a in v)) if clone else State(*v)
        return field

    def _refuse_unsupported_likelihood(self) -> None:
        """Raises for a model whose likelihood ``log_likelihood`` would get wrong."""
        if self.rc.n_lat:
            raise NotImplementedError("log_likelihood does not cover the model_conf.edep_ae latent token.")
        if self.rc.edep_whiten:
            raise NotImplementedError("log_likelihood under model_conf.edep_whiten would miss the whitening's Jacobian")

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
        """Applies ``fn`` to one chunk, tiled up to the smallest bucket that holds it and cut back after."""
        n = tensors[0].shape[dim]
        target = next((b for b in sorted(buckets or ()) if b >= n), None)
        if target is None or target == n:
            return fn(*tensors)
        # tile the chunk's own rows: valid inputs, discarded after the solve
        reps = -(-target // n)
        padded = [torch.cat([t] * reps, dim).narrow(dim, 0, target) for t in tensors]
        return fn(*padded).narrow(cat_dim, 0, n)


# ---- plumbing ------------------------------------------------------------------------------------------------------

def _time_grid(time_grid, step_size: float, method: str, reverse: bool, like: Tensor) -> Tensor:
    """Builds the solve's time grid: ``round(1 / step_size)`` equal steps over [0, 1], flipped for ``reverse``.

    An explicit ``time_grid`` (e.g. a YAML list) is used as given; ``reverse`` only flips the built one.
    """
    if time_grid is not None:
        return torch.as_tensor(time_grid, dtype=like.dtype, device=like.device)
    n    = max(round(1.0 / step_size), 1) if method in _STEPPED else 1   # an ODESolver method: the bare [0, 1]
    grid = torch.linspace(0.0, 1.0, n + 1, device=like.device, dtype=like.dtype)
    return grid.flip(0) if reverse else grid


def _no_latent(x: Tensor) -> Tensor:
    """Returns the empty latent (B, 0) of a model without one, which the steps carry along."""
    return x.new_zeros(len(x), 0)


def _as_returned(s: State, path: list, stacked: bool, latent: bool):
    """Returns ``_steps``' result: the final state or the stack of ``path``, paired with the latent if there is one."""
    if stacked:
        s = State(torch.stack([p.x for p in path]), torch.stack([p.z for p in path]))
    return tuple(s) if latent else s.x


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


def _on_every_state(f, sols: Tensor) -> Tensor:
    """Applies ``f`` to the state (B, L, C), or to every state of a ``return_timesteps`` stack (T, B, L, C)."""
    return torch.stack([f(x) for x in sols]) if sols.dim() == 4 else f(sols)


def _with_pdgids(sols: Tensor, pdgids: Tensor) -> Tensor:
    """Appends the pdgids as the last channel, on every ``return_timesteps`` state."""
    if sols.dim() == 4:
        pdgids = pdgids.unsqueeze(0).expand(sols.shape[0], -1, -1, -1)
    return torch.cat((sols, pdgids), dim=-1)


def _ode_solver(model, method: str) -> ODESolver:
    """Wraps ``model`` in flow_matching's ``ODESolver``; rk4 clones each velocity (CUDA graphs reuse the buffer)."""
    vm = (lambda *a, **kw: model(*a, **kw).clone()) if method == "rk4" else model
    return ODESolver(velocity_model=vm)


@contextlib.contextmanager
def _field_sees_graph(model, after: bool | None = None):
    """Sets ``model.no_detach`` for the block: the field sees x with its graph (inert under ``no_grad``).

    Afterwards the flag goes back to what it was, or to ``after`` when that is given.
    """
    old, model.no_detach = model.no_detach, True
    try:
        yield
    finally:
        model.no_detach = old if after is None else after
