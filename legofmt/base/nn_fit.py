"""How the base head is fitted: its loss (``head_nll``), how that scores one datum under each distribution
(``energy_logp``, ``angles_logp``, ``edep_logp``), and the recipe that minimises it (``fit_head``).

The densities themselves are ``base_dists.py``. What a fit adds to them is here, with every number it uses: data
held inside each density's open domain, float64, a uniform outlier component, and angles too small to resolve
scored as a censored mass.
"""

import torch
from torch import Tensor, nn

from legofmt.base_dist.base_dists import Angles, BaseParams, Edep, Energy
from legofmt.base_dist.nn import ieee_fp32
from legofmt.data.struct import DataStruct

# Weight of a uniform outlier component in the energy and angle densities; without it far-tail units dominate a fit.
_OUTLIER = 1e-3
# Smallest w that float32 directions resolve; a smaller angle is scored as the censored mass P(w < W0).
_W0 = 3e-4
# Where data is held, inside each density's open domain: u, w, and E / e_dep_max.
_U_RANGE = (1e-6, 1 - 1e-12)
_W_RANGE = (_W0, 1 - 1e-7)
_E_RANGE = (1e-4, 1 - 1e-4)


def head_nll(ds_t: DataStruct, fwd: Tensor, p: BaseParams, e_dep_max: float) -> Tensor:
    """The head's loss: per-unit mean NLLs (3,), float64, of the forward events' data under the per-slot prior ``p``.
    Energy per continuum slot (u > 0; the atom at 0 is the flow's), both angles per outgoing slot (momentum and
    position, each about the incoming one), E_dep per deposit event."""
    f         = ds_t.f
    slot      = ds_t.am.out_p & fwd.unsqueeze(-1)   # the outgoing particles of forward events
    continuum = slot & (f.out_energy > 0)
    deposit   = fwd & (f.edep > 0)
    lp_en     = energy_logp(p.energy, f.out_energy)
    lp_ang    = (angles_logp(p.angles, angle_w(f.out_mom, f.in_mom))
                 + angles_logp(p.angles, angle_w(f.out_pos, f.in_pos)))
    lp_edep   = edep_logp(p.edep, f.edep.unsqueeze(-1), e_dep_max).squeeze(-1)
    sums      = torch.stack(((lp_en * continuum).sum(), (lp_ang * slot).sum(), (lp_edep * deposit).sum().double()))
    counts    = torch.stack((continuum.sum(), slot.sum(), deposit.sum())).double().clamp(min=1)
    return -sums / counts


def energy_logp(dist: Energy, u: Tensor, eps: float = _OUTLIER) -> Tensor:
    """``log p(u)`` as a fit scores it: mixed with an ``eps`` uniform outlier."""
    u = u.double().clamp(*_U_RANGE)
    return _with_outlier(_float64(dist).log_density(u), eps)


def angles_logp(dist: Angles, w: Tensor, eps: float = _OUTLIER) -> Tensor:
    """``log p(w)`` as a fit scores it: mixed with an ``eps`` uniform outlier, and for ``w < W0`` the censored
    mass ``P(w < W0)`` in its place. Float32 directions round such an angle to 0, and exact zeros would
    otherwise reward ever larger pw."""
    w, dist   = w.double(), _float64(dist)
    log_p     = _with_outlier(dist.log_density(w.clamp(*_W_RANGE)), eps)
    mass0     = dist.cdf(w.new_tensor(_W0))
    log_mass0 = torch.log((1 - eps) * mass0 + eps * _W0)   # the outlier's share of [0, W0) is eps W0
    return torch.where(w < _W0, log_mass0, log_p)


def edep_logp(dist: Edep, edep: Tensor, e_dep_max: float) -> Tensor:
    """``log p(x)`` of a deposit as a fit scores it."""
    frac = (edep.float() / e_dep_max).clamp(*_E_RANGE)
    return dist.log_density(torch.logit(frac))


def angle_w(dirs: Tensor, axis: Tensor) -> Tensor:
    """``w`` of unit directions ``dirs`` (B, n, 3) about each event's ``axis`` (B, 3)."""
    axis = torch.nn.functional.normalize(axis, dim=-1).unsqueeze(1)
    cos  = (dirs * axis).sum(-1).clamp(-1, 1)   # rounding can push a product of unit vectors past +-1
    return torch.acos(cos) / torch.pi


def _with_outlier(log_p: Tensor, eps: float) -> Tensor:
    """``log((1 - eps) p + eps)``: the density ``p`` on (0, 1) mixed with a uniform one."""
    eps = log_p.new_tensor(eps)
    return torch.logaddexp(log_p + torch.log1p(-eps), torch.log(eps))


def _float64(dist):
    return type(dist)(*(v.double() for v in dist))


def fit_head(head: nn.Module, nll_of, batches, steps: int, lr: float | None = None) -> float:
    """Fit ``head`` and freeze it. ``nll_of(batch)`` is the batch's loss, or None to skip the batch.

    Recipe (2026-11 study): Adam, grad clip 1, IEEE fp32, and at step k the learning rate

    ```
    lr_k = lr * min(1, (k + 1) / warm) * (0.01 + 0.99 * (1 + cos(pi k / steps)) / 2),    warm = steps // 50
    ```

    a linear warmup times a cosine from ``lr`` (default 3e-3) down to lr / 100. A non-finite loss raises. Returns
    the final NLL."""
    opt  = torch.optim.Adam(head.parameters(), lr=3e-3 if lr is None else lr)
    warm = max(1, steps // 50)

    def lr_factor(step: int) -> float:
        cosine = torch.cos(torch.tensor(torch.pi * min(step, steps) / steps)).item()
        return min(1.0, (step + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + cosine))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    final = float("nan")
    with ieee_fp32():   # the backward too (the head's forward pins itself)
        for batch in batches:
            loss = nll_of(batch)
            if loss is None:
                continue
            if not torch.isfinite(loss):  # never freeze a NaN prior into the flow
                raise RuntimeError(f"base_head_lik pretraining: non-finite loss {loss.item()}.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            final = loss.item()
    head.requires_grad_(False)
    return final
