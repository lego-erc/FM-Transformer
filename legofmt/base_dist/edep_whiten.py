"""The flow carries the E_dep token in units of the event's own prior.

```
whiten   y = (logit(E / e_dep_max) - mu) / sig
back     E = e_dep_max sigmoid(mu + sig y)
```

``(mu, sig)`` is the event's E_dep prior from the frozen base head (``base_dists.py``). In this unit the prior is
N(0, 1), which is what the base draws (``GenerateBase.insert_add``): the base needs no map, only the data is whitened
on its way in and the samples are mapped back on their way out.

An event without deposit (E <= 0) has no logit. It sits at ``y = -6``, and anything below ``-4.5`` maps back to
E = 0; a real deposit is held at ``-4.4`` or above, so it never turns into a pass-through.

A token that is not whitened, and the options on the unit, are in :mod:`legofmt.legacy.edep_token`.
"""

import torch
from torch import Tensor

from legofmt.data.struct import _F


def whiten_edep(f: Tensor, generated: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float,
                inverse: bool = False) -> Tensor:
    """A copy of ``f`` with the E_dep token whitened (``inverse``: mapped back) where ``generated`` (B,) says the flow
    generates it. ``mu``, ``sig`` (B,); ``f`` may carry a leading time axis."""
    e = _F(f).edep
    if inverse:
        back = e_dep_max * torch.sigmoid(mu + sig*e)
        new  = torch.where(e > -4.5, back, torch.zeros_like(e))     # from below -4.5: no deposit
    else:
        frac = (e / e_dep_max).clamp(1e-6, 1 - 1e-6)
        y    = (torch.logit(frac) - mu) / sig
        new  = torch.where(e > 0, y.clamp_min(-4.4), -6.0)          # no deposit: -6; a deposit: >= -4.4
    f = f.clone()                              # e still reads the caller's tensor
    _F(f).edep.copy_(torch.where(generated, new, e))
    return f
