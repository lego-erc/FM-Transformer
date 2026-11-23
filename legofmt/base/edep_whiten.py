"""The flow carries the E_dep token in units of the event's own prior.

```
whiten   y = (logit(E / e_dep_max) - mu) / sig
back     E = e_dep_max sigmoid(mu + sig y)
```

``(mu, sig)`` is the event's E_dep prior from the frozen base head (``base_dists.py``). In this unit the prior is
N(0, 1), which is what the base samples (``GenerateBase``). So the base needs no map: only the data is whitened on
its way into the flow, and the flow's samples are mapped back on their way out.

An event without deposit (E <= 0) has no logit. Its token is set to ``y = -6``, and any ``y`` below ``-4.5`` maps
back to E = 0. A real deposit is held at ``y = -4.4`` or above, so it never maps back to E = 0.

A token that is not whitened, and the options on the unit, are in :mod:`legofmt.legacy.edep_token`.
"""

import torch
from torch import Tensor

from legofmt.data.struct import _F

_Y_NO_DEPOSIT = -6.0   # the token's y for an event without deposit
_Y_SPLIT      = -4.5   # a y below maps back to E = 0
_Y_MIN        = -4.4   # a real deposit's y is held at or above
_FRAC_EPS     = 1e-6   # E / e_dep_max is held in [eps, 1 - eps], inside the logit's open domain


def whiten(E: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float) -> Tensor:
    """Whitens a deposit ``E``: ``y = (logit(E / e_dep_max) - mu) / sig``, held at -4.4 or above; -6 without deposit."""
    y = (torch.logit(E / e_dep_max, eps=_FRAC_EPS) - mu) / sig
    return torch.where(E > 0, y.clamp(min=_Y_MIN), _Y_NO_DEPOSIT)


def unwhiten(y: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float) -> Tensor:
    """Maps ``y`` back to a deposit: ``E = e_dep_max sigmoid(mu + sig y)``, 0 for ``y <= -4.5``."""
    E = e_dep_max * torch.sigmoid(mu + sig*y)
    return torch.where(y > _Y_SPLIT, E, 0.0)


def whiten_edep(f: Tensor, generated: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float,
                inverse: bool = False) -> Tensor:
    """Whitens the E_dep token of ``f`` (``inverse``: maps it back) on the events whose E_dep the flow generates.

    Args:
        f:         (B, L, D), the tokens; may carry a leading time axis.
        generated: (B,) bool, the events whose E_dep the flow generates.
        mu, sig:   (B,), each event's E_dep prior.
        e_dep_max: the deposit's cap (``base_conf.e_dep_max``), on the scale of the token before whitening.
        inverse:   map ``y`` back to a deposit ``E``.

    Returns:
        A copy of ``f`` with the E_dep token mapped.
    """
    e   = _F(f).edep
    new = unwhiten(e, mu, sig, e_dep_max) if inverse else whiten(e, mu, sig, e_dep_max)
    f   = f.clone()   # e stays a view of the caller's tensor
    _F(f).edep.copy_(torch.where(generated, new, e))
    return f
