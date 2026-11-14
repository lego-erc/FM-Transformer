"""The flow carries the E_dep token in units of the event's own prior.

```
whiten   y = (logit(E / e_dep_max) - mu) / sig
back     E = e_dep_max sigmoid(mu + sig y)
```

``(mu, sig)`` is the event's E_dep prior from the frozen base head (``base_dists.py``). In this unit the prior is
N(0, 1), which is what the base draws (``GenerateBase``): the base needs no map, only the data is whitened
on its way in and the samples are mapped back on their way out.

An event without deposit (E <= 0) has no logit. It sits at ``y = -6``, and anything below ``-4.5`` maps back to
E = 0; a real deposit is held at ``-4.4`` or above, so it never turns into a pass-through.

A token that is not whitened, and the options on the unit, are in :mod:`legofmt.legacy.edep_token`.
"""

import torch
from torch import Tensor

from legofmt.data.struct import _F

# The token's values y for an event without deposit, and what holds E / e_dep_max inside the logit's open domain.
_Y_NO_DEPOSIT = -6.0
_Y_SPLIT      = -4.5   # a sample below maps back to E = 0
_Y_MIN        = -4.4   # a real deposit is held at or above
_FRAC_RANGE   = (1e-6, 1 - 1e-6)


def whiten_edep(f: Tensor, generated: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float,
                inverse: bool = False) -> Tensor:
    """Whitens the E_dep token of ``f`` (``inverse``: maps it back) on the events whose E_dep the flow generates.

    Args:
        f:         (B, N, D) tokens; may carry a leading time axis.
        generated: (B,) bool, the events whose E_dep the flow generates.
        mu, sig:   (B,), each event's E_dep prior.
        e_dep_max: the deposit's cap.
        inverse:   map ``y`` back to a deposit ``E``.

    Returns:
        A copy of ``f``.
    """
    e = _F(f).edep
    if inverse:
        back = e_dep_max * torch.sigmoid(mu + sig*e)
        new  = torch.where(e > _Y_SPLIT, back, torch.zeros_like(e))
    else:
        frac = (e / e_dep_max).clamp(*_FRAC_RANGE)
        y    = (torch.logit(frac) - mu) / sig
        new  = torch.where(e > 0, y.clamp_min(_Y_MIN), _Y_NO_DEPOSIT)
    f = f.clone()   # e still reads the caller's tensor
    _F(f).edep.copy_(torch.where(generated, new, e))
    return f
