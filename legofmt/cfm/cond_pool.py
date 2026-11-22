"""The cond scalars as one token of the field: a set encoder over their values (``model_args.cond_pool``).

```
z_k    = (value_k - mu_k) / sd_k                 mu, sd: the training data's, per scalar
token  = rho(sum_k phi([z_k, onehot_k]))         phi, rho: two-layer SiLU MLPs of the field's width
```

One token instead of one per scalar, so the encoder runs ``n - 1`` tokens fewer per event. The sum makes the token a
function of the set of (which scalar, value) pairs. ``z`` enters as it is: Fourier features of it broke the
interpolation between trained densities (A/B of 20.10.2026, where this form matched the per-scalar tokens).
"""

import torch
from torch import Tensor, nn


class CondPool(nn.Module):
    """Pools the ``n`` cond scalars of an event into one token of width ``h_dim`` (module docstring)."""

    def __init__(self, n: int, h_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1 + n, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))
        self.rho = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sd", torch.ones(n))

    def standardise_by(self, values: Tensor) -> None:
        """Sets ``mu`` and ``sd`` to those of ``values`` (N, n), the training events' scalars.

        A scalar the dataset holds fixed keeps ``sd = 1``: off its one value it then moves in its own unit.
        """
        sd = values.std(0)
        self.mu.copy_(values.mean(0))
        self.sd.copy_(torch.where(sd > 1e-6, sd, 1.0))

    def forward(self, values: Tensor) -> Tensor:
        """Returns the token (B, 1, h_dim) of the scalars ``values`` (B, n)."""
        z     = (values - self.mu) / self.sd
        which = torch.eye(z.shape[1], dtype=z.dtype, device=z.device).expand(len(z), -1, -1)
        return self.rho(self.phi(torch.cat((z.unsqueeze(-1), which), dim=-1)).sum(1)).unsqueeze(1)
