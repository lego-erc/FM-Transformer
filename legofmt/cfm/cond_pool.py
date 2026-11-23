"""The conditioning scalars as one token of the field: a set encoder over their values (``model_args.cond_pool``).

```
z_k    = (value_k - mu_k) / sd_k                 mu_k, sd_k: scalar k's mean and std over the training events
token  = rho(sum_k phi([z_k, onehot_k]))         phi, rho: two-layer SiLU MLPs of the field's width
```

The ``n`` conditioning scalars enter the field as one token instead of one each, so its encoder runs ``n - 1`` tokens
fewer per event. The sum makes the token a function of the set of (which scalar, value) pairs. ``z_k`` enters as it
is: Fourier features of it break the interpolation between the trained densities.
"""

import torch
from torch import Tensor, nn


class CondPool(nn.Module):
    """The set encoder that pools an event's ``n`` conditioning scalars into one token of width ``h_dim``."""

    def __init__(self, n: int, h_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1 + n, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))
        self.rho = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sd", torch.ones(n))

    def standardise_by(self, values: Tensor) -> None:
        """Sets ``mu`` and ``sd`` to the mean and std of ``values`` (N, n), the training events' conditioning scalars.

        A scalar that is constant over the dataset keeps ``sd = 1``: a value off the trained one then enters as their
        difference, in the scalar's own unit.
        """
        sd = values.std(0)
        self.mu.copy_(values.mean(0))
        self.sd.copy_(torch.where(sd > 1e-6, sd, 1.0))

    def forward(self, values: Tensor) -> Tensor:
        """Returns the pooled token (B, 1, h_dim) of the conditioning scalars ``values`` (B, n)."""
        z     = (values - self.mu) / self.sd
        which = torch.eye(z.shape[1], dtype=z.dtype, device=z.device).expand(len(z), -1, -1)
        return self.rho(self.phi(torch.cat((z.unsqueeze(-1), which), dim=-1)).sum(1)).unsqueeze(1)
