"""Per-layer E_dep autoencoder (``model_conf.edep_ae``): K latents per event for the flow, and a factorised
stochastic decoder back to the n_layers deposits.

The flow generates the K standardised latents as one extra token of the field (``model_args.lat_dim``).
After the solve, ``sample`` draws every layer from its own zero-deposit probability and log-E Gaussian mixture,
given the latents and the event's total, and rescales the draw to that total. A deterministic decoder cannot do
this: what the latents leave out is per-layer sampling noise, uncorrelated across layers, and reconstructing it
put the Si layers of stack59 at 0.8x their mean and 0.5x their spread at every K. Each draw is clamped at the
layer's training maximum (single far-tail draws otherwise steal the total on rescaling) and an all-zero draw gives
the total to the likeliest layer. All energies in MeV.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOG_OFFSET = 0.1               # MeV, the encoder's log(E + offset): about a MIP in 0.3 mm Si
_HALF_LOG_2PI = 0.9189385332046727


def _mlp(n_in: int, n_out: int, hidden: int, depth: int) -> nn.Sequential:
    dims = [n_in] + [hidden] * depth
    return nn.Sequential(*(m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.GELU())),
                         nn.Linear(hidden, n_out))


class EdepLayerAE(nn.Module):
    def __init__(self, n_layers: int, K: int, hidden: int = 512, n_mix: int = 8) -> None:
        super().__init__()
        self.n_layers, self.K, self.n_mix = n_layers, K, n_mix
        self.enc = _mlp(n_layers, K, hidden, 2)
        self.dec = _mlp(K + 1, n_layers * (1 + 3 * n_mix), hidden, 3)
        # fitted after training (scripts/train_edep_ae.py): latent standardisation, per-layer draw ceiling
        self.register_buffer("z_mu", torch.zeros(K))
        self.register_buffer("z_sd", torch.ones(K))
        self.register_buffer("log_e_max", torch.full((n_layers,), float("inf")))

    def encode(self, E: Tensor) -> Tensor:
        """(B, n_layers) MeV -> (B, K) standardised latents."""
        return (self.enc(torch.log(E + LOG_OFFSET)) - self.z_mu) / self.z_sd

    def _head(self, z: Tensor, tot: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        M = self.n_mix
        h = self.dec(torch.cat([z * self.z_sd + self.z_mu, torch.log(tot)], -1))
        h = h.view(-1, self.n_layers, 1 + 3 * M)
        return h[..., 0], F.log_softmax(h[..., 1:1 + M], -1), h[..., 1 + M:1 + 2 * M], h[..., 1 + 2 * M:].clamp(-7, 3)

    def nll(self, E: Tensor) -> Tensor:
        """Mean per-layer negative log-likelihood of (B, n_layers) deposits; every event needs a deposit."""
        a, lw, mu, ls = self._head(self.encode(E), E.sum(-1, keepdim=True))
        v = torch.log(E.clamp(min=1e-12))[..., None]
        logn = -0.5 * ((v - mu) / ls.exp()) ** 2 - ls - _HALF_LOG_2PI
        cont = torch.logsumexp(lw + logn, -1) - v[..., 0]          # density of E, not of log E
        return -torch.where(E <= 0, F.logsigmoid(a), F.logsigmoid(-a) + cont).mean()

    @torch.no_grad()
    def sample(self, z: Tensor, tot: Tensor) -> Tensor:
        """(B, K) standardised latents, (B,) totals [MeV] -> (B, n_layers) deposits that sum to the totals."""
        z, tot = z.float(), tot.float().reshape(-1, 1)
        a, lw, mu, ls = self._head(z, tot.clamp(min=1e-6))
        k = torch.distributions.Categorical(logits=lw).sample()[..., None]
        v = mu.gather(-1, k)[..., 0] + ls.gather(-1, k)[..., 0].exp() * torch.randn_like(a)
        E = torch.where(torch.rand_like(a) < torch.sigmoid(a), torch.zeros_like(a),
                        torch.minimum(v, self.log_e_max).exp())
        dead = E.sum(-1) == 0
        E[dead, a[dead].argmin(-1)] = 1.0
        return torch.where(tot > 0, E * tot / E.sum(-1, keepdim=True), torch.zeros_like(E))


def build_edep_ae(spec: dict) -> EdepLayerAE:
    """``spec`` = ``{"config", "state_dict"}`` (scripts/train_edep_ae.py's output, which a flow checkpoint stashes
    in ``model_conf.edep_ae``); frozen."""
    ae = EdepLayerAE(**spec["config"])
    ae.load_state_dict(spec["state_dict"])
    ae.requires_grad_(False)
    return ae.eval()
