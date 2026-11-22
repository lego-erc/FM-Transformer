"""K summary tokens of a layered stack, in the flow's hidden width (``model_args.geom_encoder``).

```
layer      [t mm, rho g/cm3, Z, A], beam side first

features   log t, log rho, Z / 100, log A, log(t / X0), log(t / lambda_I), log(rho t)          the material
           sin, cos of 2 pi c_X0 / p and of 2 pi c_lam / (p / 10)                             the depth c of the
           c_X0 / 30, c_lam / 3                                                               layer's centre

encoder    h      = Encoder([summary_1 .. summary_K | inp(features)])
           tokens = out(h_1 .. h_K)
```

``c_X0`` / ``c_lam`` is the cumulative depth in radiation / interaction lengths: physical depth rather than the layer
index, so stacks of different layer counts line up; ``p`` are ``N_FREQ`` periods from 0.2 to 60 X0. The summary
tokens are learned, the encoder is a small bidirectional x-transformers ``Encoder``, and ``out`` starts at zero, so
an untrained encoder adds nothing.

The stack is fixed per model (it comes from the config), so the layer features are a non-persistent buffer and the
encoder runs once per vector-field call on a batch of one.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from x_transformers import Encoder

from legofmt.geometry.material import lambda_mm, x0_mm

N_FREQ = 8


def layer_features(layers: Tensor) -> Tensor:
    """Builds the encoder's features (Lg, 7 + 4 * N_FREQ + 2) from the layers (Lg, 4) = [t, rho, Z, A]."""
    t, rho, Z, A = layers.unbind(-1)
    t_x0   = t / x0_mm(rho, Z, A)          # the layer's thickness in radiation lengths
    t_lam  = t / lambda_mm(rho, A)         # and in interaction lengths
    c_x0   = t_x0.cumsum(0) - t_x0 / 2     # the depth of the layer's centre
    c_lam  = t_lam.cumsum(0) - t_lam / 2
    period = _periods(layers.dtype)
    ph_x0  = 2 * torch.pi * c_x0.unsqueeze(1) / period            # (Lg, N_FREQ)
    ph_lam = 2 * torch.pi * c_lam.unsqueeze(1) / (period / 10)
    material = [torch.log(t), torch.log(rho), Z / 100, torch.log(A),
                torch.log(t_x0), torch.log(t_lam), torch.log(rho * t)]
    depth    = [torch.sin(ph_x0), torch.cos(ph_x0), torch.sin(ph_lam), torch.cos(ph_lam), c_x0 / 30, c_lam / 3]
    return torch.column_stack((*material, *depth))


class GeomEncoder(nn.Module):
    """The encoder of a layered stack into ``tokens`` summary tokens of width ``h_dim``."""

    def __init__(self, layers, h_dim: int, *, tokens: int = 4, dim: int = 128, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        feats = layer_features(torch.as_tensor(layers, dtype=torch.float64)).float()
        self.register_buffer("feats", feats.unsqueeze(0), persistent=False)
        self.K = tokens
        self.inp = nn.Linear(feats.shape[-1], dim)
        self.summary = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.enc = Encoder(dim=dim, depth=depth, heads=heads)
        self.out = nn.Linear(dim, h_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self) -> Tensor:
        """Encodes the geometry into its summary tokens (1, K, h_dim)."""
        h = torch.cat((self.summary.unsqueeze(0), self.inp(self.feats)), dim=1)
        return self.out(self.enc(h)[:, :self.K])


# ---- plumbing ------------------------------------------------------------------------------------------------------

def _periods(dtype) -> Tensor:
    """Returns the ``N_FREQ`` periods of the depth features, log-spaced from 0.2 to 60."""
    return torch.logspace(*torch.log10(torch.tensor([0.2, 60.0], dtype=torch.float64)).tolist(), N_FREQ, dtype=dtype)
