"""Per-layer E_dep autoencoder (``model_conf.edep_ae``): K latents per event for the flow, and a factorised
stochastic decoder back to the n_layers deposits.

```
encode    z = (enc(log(E + offset)) - z_mu) / z_sd            E: the deposits per layer [MeV], z: K latents

decode    a_l, (log w, mu, log sig)_lm = dec(z, log E_tot)    per layer l and mixture component m
          P(E_l = 0)             = sigmoid(a_l)
          p(log E_l | E_l > 0)   = sum_m w_lm N(mu_lm, sig_lm^2)
```

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

from legofmt.rng import AE_MIX, AE_NORM, AE_ZERO

LOG_OFFSET = 0.1               # MeV, the encoder's log(E + offset): about a MIP in 0.3 mm Si
_HALF_LOG_2PI = 0.9189385332046727


def _mlp(n_in: int, n_out: int, hidden: int, depth: int) -> nn.Sequential:
    dims = [n_in] + [hidden] * depth
    return nn.Sequential(*(m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.GELU())),
                         nn.Linear(hidden, n_out))


class EdepLayerAE(nn.Module):
    """The autoencoder: an MLP encoder to K latents and an MLP decoder to each layer's mixture."""

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
        """Encodes per-layer deposits (B, n_layers) [MeV] into standardised latents (B, K)."""
        return (self.enc(torch.log(E + LOG_OFFSET)) - self.z_mu) / self.z_sd

    def _head(self, z: Tensor, tot: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decodes latents (B, K) and totals (B, 1) into each layer's ``(a, log w, mu, log sig)``.

        ``a`` is (B, n_layers), the mixture's parameters (B, n_layers, n_mix); ``log sig`` is held in [-7, 3].
        """
        M = self.n_mix
        h = self.dec(torch.cat([z * self.z_sd + self.z_mu, torch.log(tot)], -1))
        h = h.view(-1, self.n_layers, 1 + 3 * M)
        return h[..., 0], F.log_softmax(h[..., 1:1 + M], -1), h[..., 1 + M:1 + 2 * M], h[..., 1 + 2 * M:].clamp(-7, 3)

    def nll(self, E: Tensor) -> Tensor:
        """Returns the mean per-layer negative log-likelihood of deposits (B, n_layers).

        Every event needs a deposit.
        """
        a, log_w, mu, log_sig = self._head(self.encode(E), E.sum(-1, keepdim=True))
        log_e = torch.log(E.clamp(min=1e-12))[..., None]
        log_n = -0.5 * ((log_e - mu) / torch.exp(log_sig)) ** 2 - log_sig - _HALF_LOG_2PI
        log_p = torch.logsumexp(log_w + log_n, -1) - log_e[..., 0]          # density of E, not of log E
        return -torch.where(E <= 0, F.logsigmoid(a), F.logsigmoid(-a) + log_p).mean()

    @torch.no_grad()
    def sample(self, z: Tensor, tot: Tensor, rng=None) -> Tensor:
        """Samples per-layer deposits that sum to each event's total.

        Args:
            z:   (B, K) standardised latents.
            tot: (B,) totals [MeV].
            rng: keys the draws per event and layer (``legofmt.rng.EventRNG``); None: torch's noise.

        Returns:
            (B, n_layers) deposits [MeV].
        """
        z, tot = z.float(), tot.float().reshape(-1, 1)
        a, log_w, mu, log_sig = self._head(z, tot.clamp(min=1e-6))
        layer = torch.arange(a.shape[-1], device=a.device)
        # the noise: a mixture component, a normal and a uniform per layer
        k = (torch.distributions.Categorical(logits=log_w).sample() if rng is None
             else rng.categorical(log_w.softmax(-1), AE_MIX, layer))[..., None]
        n = torch.randn_like(a) if rng is None else rng.normal(AE_NORM, layer).to(a)
        u = torch.rand_like(a) if rng is None else rng.uniform(AE_ZERO, layer).to(a)

        log_e = mu.gather(-1, k)[..., 0] + torch.exp(log_sig.gather(-1, k)[..., 0]) * n
        E     = torch.where(u < torch.sigmoid(a), 0.0, torch.exp(torch.minimum(log_e, self.log_e_max)))
        dead  = E.sum(-1, keepdim=True) == 0      # nothing drawn anywhere: the total goes to the likeliest layer
        E     = torch.where(dead, E.scatter(-1, a.argmin(-1, keepdim=True), 1.0), E)
        return torch.where(tot > 0, E * tot / E.sum(-1, keepdim=True), 0.0)


def build_edep_ae(spec: dict) -> EdepLayerAE:
    """Constructs the per-layer E_dep autoencoder from ``spec``, frozen.

    Args:
        spec: ``{"config", "state_dict"}``, scripts/train_edep_ae.py's output, which a flow checkpoint stashes in
            ``model_conf.edep_ae``.

    Returns:
        The autoencoder, frozen and in eval mode.
    """
    ae = EdepLayerAE(**spec["config"])
    ae.load_state_dict(spec["state_dict"])
    ae.requires_grad_(False)
    return ae.eval()
