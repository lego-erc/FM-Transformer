"""A material's two length scales, from its density ``rho`` [g/cm3], charge ``Z`` and mass number ``A``.

```
X0        = 716.4 A / (Z (Z + 1) log(287 / sqrt(Z)))  g/cm2      the radiation length (Dahl's fit)
lambda_I  = 35 A**(1/3)                               g/cm2      the nuclear interaction length (PDG approximation)
in mm     = 10 [g/cm2] / rho
```

The geometry encoder measures a layer's thickness in them (``cfm/geom_encoder.py``), the base head a block's
(``base/interface.py``).
"""

import torch
from torch import Tensor


def x0_gcm2(Z: Tensor, A: Tensor) -> Tensor:
    """Computes the radiation length (Dahl's fit), in g/cm2."""
    return 716.4 * A / (Z * (Z + 1) * torch.log(287.0 / torch.sqrt(Z)))


def x0_mm(rho: Tensor, Z: Tensor, A: Tensor) -> Tensor:
    """Computes the radiation length, in mm."""
    return x0_gcm2(Z, A) / rho * 10.0


def lambda_mm(rho: Tensor, A: Tensor) -> Tensor:
    """Computes the nuclear interaction length, lambda_I ~ 35 A^(1/3) g cm^-2 (PDG approximation), in mm."""
    return 35.0 * A ** (1.0 / 3.0) / rho * 10.0
