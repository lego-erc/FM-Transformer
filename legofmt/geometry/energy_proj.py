"""``EnergyProjections``: MeV <-> the model's energy scalars.

```
incoming   e     = log(E / cutoff) / log(max_energy / cutoff)             held in [0, 1]
outgoing   e_out = 1 - log(E_out / cutoff) / log(E_in / cutoff)           inverted, and relative to the incoming
                                                                          particle (written by DataPrep.cc_trafo)
back       E     = cutoff (max_energy / cutoff) ** e
           E_out = cutoff (max_energy / cutoff) ** ((1 - e_out) e)        e: the incoming scalar
```

The pre-token 6-column transforms (``norm_type``) are :mod:`legofmt.legacy.energy_proj`.
"""

import torch
from torch import Tensor

from legofmt.legacy.energy_proj import LegacyEnergyTransforms


class EnergyProjections(LegacyEnergyTransforms):
    """The energy scales of one model.

    Args:
        norm_type:  legacy: the pre-token transform a call applies.
        cutoff_mev: the dataset's energy cutoff [MeV].
        max_energy: the top of the incoming scale [MeV].
    """

    def __init__(self, norm_type: str | bool = "in_frac", cutoff_mev: float = 10.0, max_energy: float | None = None):
        super().__init__(norm_type)
        self.cutoff = cutoff_mev
        self.max_energy = max_energy
        self.log_range = torch.log(torch.tensor(max_energy / cutoff_mev)).item() if max_energy else None

    def to_scalar(self, mom: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        """Splits a momentum [MeV] into its unit direction and the incoming scalar of its magnitude."""
        norm = mom.norm(dim=-1, keepdim=True)
        return mom / norm.clamp_min(eps), self.to_scalar_e(norm, eps)

    def to_scalar_e(self, e_mev: Tensor, eps: float = 1e-8) -> Tensor:
        """Converts an energy [MeV] to the incoming scalar, held in [0, 1]."""
        return (torch.log(e_mev.clamp_min(eps) / self.cutoff) / self.log_range).clamp(0.0, 1.0)

    def from_scalar(self, dir_: Tensor, e: Tensor) -> Tensor:
        """Inverts ``to_scalar``: the incoming scalar and a unit direction -> the momentum [MeV]."""
        return dir_ * (self.cutoff * (self.max_energy / self.cutoff) ** e.clamp(0.0, 1.0))

    def to_mev(self, e_out: Tensor, e_in: Tensor) -> Tensor:
        """Converts an outgoing scalar to MeV, given its event's incoming scalar ``e_in``."""
        return self.cutoff * (self.max_energy / self.cutoff) ** ((1 - e_out.clamp(0.0, 1.0)) * e_in.clamp(0.0, 1.0))
