"""How the base head is fitted: its loss (``head_nll``), how that scores one datum under each distribution
(``energy_logp``, ``angles_logp``, ``edep_logp``), and the optimisation that minimises it (``fit_head``).

The densities themselves are ``base_dists.py``. What a fit adds to them is here, with every number it uses: data
held inside each density's open domain, float64, a uniform outlier component, and angles too small to resolve
scored as a censored mass.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from legofmt.base.base_dists import Angles, BaseParams, Edep, Energy
from legofmt.base.nn import ieee_fp32
from legofmt.data.struct import DataStruct
from legofmt.numerics import safe_acos

# Weight of a uniform outlier component in the energy and angle densities; without it far-tail data dominates a fit.
_OUTLIER = 1e-3
# Smallest w that float32 directions resolve; a smaller angle is scored as the censored mass P(w < _W0).
_W0 = 3e-4
# Where data is held, inside each density's open domain: u, w, and E / e_dep_max.
_U_RANGE = (1e-6, 1 - 1e-12)
_W_RANGE = (_W0, 1 - 1e-7)
_E_RANGE = (1e-4, 1 - 1e-4)


def head_nll(ds_t: DataStruct, fwd: Tensor, p: BaseParams, e_dep_max: float) -> Tensor:
    """Computes the head's loss: the mean NLL of the forward events' data under their prior.

    Energy is scored on the outgoing slots with ``u > 0``: the point mass at ``u = 0`` (an outgoing particle at or
    above the incoming energy) is left to the flow. Both angles (momentum and position, each to the incoming
    particle's) are scored on every outgoing slot, E_dep on the events with a deposit.

    Args:
        ds_t:      the batch.
        fwd:       (B,) bool, the forward events.
        p:         the prior per outgoing slot (``BaseDist._slot_prior``).
        e_dep_max: the deposit's cap, the unit of ``E / e_dep_max``.

    Returns:
        (3,) float64: the mean NLL of energy, angles and E_dep, each over the slots or events it is scored on.
    """
    f         = ds_t.f
    slot      = ds_t.am.out_p & fwd.unsqueeze(-1)   # the outgoing particles of forward events
    continuum = slot & (f.out_energy > 0)
    deposit   = fwd & (f.edep > 0)
    lp_en     = energy_logp(p.energy, f.out_energy)
    lp_ang    = (angles_logp(p.angles, angle_w(f.out_mom, f.in_mom))
                 + angles_logp(p.angles, angle_w(f.out_pos, f.in_pos)))
    lp_edep   = edep_logp(p.edep, f.edep, e_dep_max)
    sums      = torch.stack(((lp_en * continuum).sum(), (lp_ang * slot).sum(), (lp_edep * deposit).sum().double()))
    counts    = torch.stack((continuum.sum(), slot.sum(), deposit.sum())).double().clamp(min=1)
    return -sums / counts


def energy_logp(dist: Energy, u: Tensor, eps: float = _OUTLIER) -> Tensor:
    """Returns ``log p(u)`` as a fit scores it: mixed with a uniform outlier of weight ``eps``.

    ``u`` is held inside ``_U_RANGE``.
    """
    u = u.double().clamp(*_U_RANGE)
    return _with_outlier(_float64(dist).log_density(u), eps)


def angles_logp(dist: Angles, w: Tensor, eps: float = _OUTLIER) -> Tensor:
    """Returns ``log p(w)`` as a fit scores it: mixed with a uniform outlier of weight ``eps``, censored below ``_W0``.

    For ``w < _W0`` the censored mass ``P(w < _W0)`` takes the density's place: float32 directions round such an
    angle to 0, where the density diverges for ``pw > 1``, so exact zeros would reward ever larger ``pw``. Any other
    ``w`` is held inside ``_W_RANGE``.
    """
    w, dist   = w.double(), _float64(dist)
    log_p     = _with_outlier(dist.log_density(w.clamp(*_W_RANGE)), eps)
    mass0     = dist.cdf(w.new_tensor(_W0))
    log_mass0 = torch.log((1 - eps) * mass0 + eps * _W0)   # the outlier's share of [0, _W0) is eps _W0
    return torch.where(w < _W0, log_mass0, log_p)


def edep_logp(dist: Edep, edep: Tensor, e_dep_max: float) -> Tensor:
    """Returns ``log p(x)`` of a deposit ``edep`` as a fit scores it, ``x = logit(edep / e_dep_max)``.

    ``edep / e_dep_max`` is held inside ``_E_RANGE``.
    """
    frac = (edep.float() / e_dep_max).clamp(*_E_RANGE)
    return dist.log_density(torch.logit(frac))


def angle_w(dirs: Tensor, axis: Tensor) -> Tensor:
    """Computes ``w = theta / pi`` (B, n) of unit directions ``dirs`` (B, n, 3) about each event's ``axis`` (B, 3)."""
    axis = F.normalize(axis, dim=-1).unsqueeze(1)
    return safe_acos((dirs * axis).sum(-1)) / torch.pi


def _with_outlier(log_p: Tensor, eps: float) -> Tensor:
    """Mixes the density ``p`` on (0, 1) with a uniform one: returns ``log((1 - eps) p + eps)``."""
    eps = log_p.new_tensor(eps)
    return torch.logaddexp(log_p + torch.log1p(-eps), torch.log(eps))


def _float64(dist):
    """Casts a distribution's parameters to float64."""
    return type(dist)(*(v.double() for v in dist))


def fit_head(head: nn.Module, nll_of, batches, steps: int, lr: float | None = None) -> float:
    """Fits ``head`` on ``batches`` and freezes it.

    Adam with the gradient norm clipped at 1, in IEEE fp32. At step ``k`` the learning rate is

    ```
    lr_k = lr * min(1, (k + 1) / warm) * (0.01 + 0.99 * (1 + cos(pi k / steps)) / 2),    warm = max(1, steps // 50)
    ```

    a linear warmup times a cosine from ``lr`` down to ``lr / 100``.

    Args:
        head:    the network to fit.
        nll_of:  ``nll_of(batch)`` is the batch's loss, or None to skip the batch.
        batches: the training batches, one step each.
        steps:   the schedule's length.
        lr:      the peak learning rate; None: 3e-3.

    Returns:
        The last scored batch's loss; NaN if no batch was scored.

    Raises:
        RuntimeError: a non-finite loss.
    """
    opt  = torch.optim.Adam(head.parameters(), lr=3e-3 if lr is None else lr)
    warm = max(1, steps // 50)

    def lr_factor(step: int) -> float:
        cosine = torch.cos(torch.tensor(torch.pi * min(step, steps) / steps)).item()
        return min(1.0, (step + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + cosine))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    final = float("nan")
    with ieee_fp32():   # for the backward: the head's forward already pins fp32 itself
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
