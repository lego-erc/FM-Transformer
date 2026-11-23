"""The pre-token energy transforms of ``EnergyProjections``: (B, N, 6) batches ``(p, x)`` of momenta and positions
with the incoming particle at index 0, a layout older than the padded token layout (they do not follow ``_F``).

Kept for lego_eval's ``EnergyProjections("in_frac_log")``; ``EnergyProjections`` derives from ``LegacyEnergyTransforms``
so that its first positional argument stays ``norm_type``.
"""

import torch
from torch import Tensor


class LegacyEnergyTransforms:
    """The pre-token transforms of a (B, N, 6) batch ``(p, x)``; ``norm_type`` names the one a call applies.

    ```
    identity      p
    log           p (1 - log|p|) / |p|            the direction of p, with length 1 - log|p|
    in_frac       p_out / |p_in|                  the incoming momentum p_in itself is kept
    in_frac_log   log of in_frac's p_out
    ```

    The positions ``x`` pass unchanged. A ``norm_type`` that is not a string selects ``identity``.
    """

    def __init__(self, norm_type: str | bool = "in_frac"):
        self.func = getattr(self, norm_type) if isinstance(norm_type, str) else self.identity

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def identity(self, p_x: Tensor) -> Tensor:
        return p_x

    def log(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat((p * (1 - p_norm.log()) / p_norm, x), -1)

    def in_frac(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        in_norm = p[:, 0:1].norm(dim=-1, keepdim=True)
        p_normed = torch.cat((p[:, :1], p[:, 1:] / in_norm), dim=1)
        return torch.cat((p_normed, x), -1)

    def in_frac_log(self, p_x: Tensor) -> Tensor:
        p_x = self.in_frac(p_x)
        p_x[:, 1:] = self.log(p_x[:, 1:])
        return p_x
