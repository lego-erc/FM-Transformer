"""K summary tokens of a layered stack, in the flow's hidden width (model_args.geom_encoder).

One token per layer, beam side first: [thickness mm, density g/cm3, Z, A] -> log t, log rho, Z/100, log A,
log(t/X0), log(t/lambda_I), log(rho t), plus Fourier features of the cumulative depth at the layer centre in X0 and
in lambda_I (physical depth rather than the layer index, so stacks of different layer counts line up). K learned
summary tokens are prepended, a small bidirectional x-transformers Encoder runs over [summary; layers], and the
summary outputs are projected to h_dim through a zero-initialised read-out, so an untrained encoder adds nothing.

The stack is fixed per model (it comes from the config), so the layer features are a non-persistent buffer and the
encoder runs once per vector-field call on a batch of one. Design and the standalone probe:
docs/superpowers/specs/2026-10-20-geometry-condition-encoder-design.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from x_transformers import Encoder

N_FREQ = 8


def x0_gcm2(Z: Tensor, A: Tensor) -> Tensor:
    """Radiation length (Dahl's fit), in g/cm2; also the base head's (base_nn.py)."""
    return 716.4 * A / (Z * (Z + 1) * torch.log(287.0 / Z.sqrt()))


def x0_mm(rho: Tensor, Z: Tensor, A: Tensor) -> Tensor:
    """Radiation length, in mm."""
    return x0_gcm2(Z, A) / rho * 10.0


def lambda_mm(rho: Tensor, A: Tensor) -> Tensor:
    """Nuclear interaction length, lambda_I ~ 35 A^(1/3) g cm^-2 (PDG approximation), in mm."""
    return 35.0 * A ** (1.0 / 3.0) / rho * 10.0


def layer_features(layers: Tensor) -> Tensor:
    """(Lg, 4) [t, rho, Z, A] -> (Lg, 7 + 4 * N_FREQ + 2)."""
    t, rho, Z, A = layers.unbind(-1)
    tx0, tlam = t / x0_mm(rho, Z, A), t / lambda_mm(rho, A)
    cx0, clam = tx0.cumsum(0) - tx0 / 2, tlam.cumsum(0) - tlam / 2
    per = torch.logspace(*torch.tensor([0.2, 60.0], dtype=torch.float64).log10().tolist(), N_FREQ, dtype=layers.dtype)
    ph_x0, ph_lam = 2 * torch.pi * cx0[:, None] / per, 2 * torch.pi * clam[:, None] / (per / 10)
    return torch.cat([torch.stack([t.log(), rho.log(), Z / 100, A.log(), tx0.log(), tlam.log(), (rho * t).log()], -1),
                      ph_x0.sin(), ph_x0.cos(), ph_lam.sin(), ph_lam.cos(), cx0[:, None] / 30, clam[:, None] / 3], -1)


class GeomEncoder(nn.Module):
    def __init__(self, layers, h_dim: int, *, tokens: int = 4, dim: int = 128, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        feats = layer_features(torch.as_tensor(layers, dtype=torch.float64)).float()
        self.register_buffer("feats", feats[None], persistent=False)
        self.K = tokens
        self.inp = nn.Linear(feats.shape[-1], dim)
        self.summary = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.enc = Encoder(dim=dim, depth=depth, heads=heads)
        self.out = nn.Linear(dim, h_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self) -> Tensor:
        """(1, K, h_dim)."""
        h = torch.cat([self.summary[None], self.inp(self.feats)], dim=1)
        return self.out(self.enc(h)[:, :self.K])
