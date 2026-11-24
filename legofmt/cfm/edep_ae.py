"""The autoencoder of the E_dep layers (``model_conf.edep_ae``): K latents per event, decoded by sampling each layer.

```
encode    z = (enc(log(E + LOG_OFFSET)) - z_mu) / z_sd                E: the deposits per layer [MeV], z: K latents

decode    a_l, (log w, mu, log sig)_lm = dec([z z_sd + z_mu, log tot])    tot: the event's total deposit [MeV];
                                                                          per layer l and mixture component m
          P(E_l = 0)             = sigmoid(a_l)
          p(log E_l | E_l > 0)   = sum_m w_lm N(log E_l; mu_lm, sig_lm)
```

The flow generates the K standardised latents as one extra token of the field (``model_args.lat_dim``).
After the solve, ``sample`` samples every layer from its own zero-deposit probability and log-E Gaussian mixture,
given the latents and the event's total, and rescales the layers to that total. A deterministic decoder cannot do
this: what the latents leave out is per-layer sampling noise, uncorrelated across layers, and a decoder that
reconstructs it puts the Si layers of stack59 at 0.8x their mean and 0.5x their spread, at every K. Each sampled
deposit is clamped at the layer's largest deposit in the training data, because a single far-tail value would take
most of the total in the rescaling. When every layer is sampled empty, the total goes to the layer least likely to
be empty. All energies are in MeV.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from legofmt.rng import AE_MIX, AE_NORM, AE_ZERO, categorical, normal_like, uniform_like

LOG_OFFSET = 0.1               # [MeV], in the encoder's log(E + LOG_OFFSET): about a MIP's deposit in 0.3 mm Si
_HALF_LOG_2PI = 0.9189385332046727


def _mlp(n_in: int, n_out: int, hidden: int, depth: int) -> nn.Sequential:
    """Builds an MLP from ``n_in`` to ``n_out``: ``depth`` hidden layers of width ``hidden``, each with a GELU."""
    dims = [n_in] + [hidden] * depth
    return nn.Sequential(*(m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.GELU())),
                         nn.Linear(hidden, n_out))


class EdepLayerAE(nn.Module):
    """The autoencoder: an MLP encoder to K latents, and an MLP decoder to each layer's zero-deposit logit and mixture.

    Both MLPs have the width ``hidden``; each layer's mixture has ``n_mix`` components.
    """

    def __init__(self, n_layers: int, K: int, hidden: int = 512, n_mix: int = 8) -> None:
        super().__init__()
        self.n_layers, self.K, self.n_mix = n_layers, K, n_mix
        self.enc = _mlp(n_layers, K, hidden, 2)
        self.dec = _mlp(K + 1, n_layers * (1 + 3 * n_mix), hidden, 3)
        # set after training (``scripts/train_edep_ae.py``): the latents' mean and std, and each layer's largest log
        # deposit in the training data
        self.register_buffer("z_mu", torch.zeros(K))
        self.register_buffer("z_sd", torch.ones(K))
        self.register_buffer("log_e_max", torch.full((n_layers,), float("inf")))

    def encode(self, E: Tensor) -> Tensor:
        """Encodes per-layer deposits (B, n_layers) [MeV] into standardised latents (B, K)."""
        return (self.enc(torch.log(E + LOG_OFFSET)) - self.z_mu) / self.z_sd

    def _head(self, z: Tensor, tot: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decodes standardised latents (B, K) and totals (B, 1) [MeV] into each layer's ``(a, log w, mu, log sig)``.

        ``a`` is (B, n_layers), the zero-deposit logit. The mixture's parameters are (B, n_layers, n_mix): ``log w``
        normalised, ``log sig`` clamped to [-7, 3].
        """
        M = self.n_mix
        h = self.dec(torch.cat((z * self.z_sd + self.z_mu, torch.log(tot)), dim=-1))
        a, log_w, mu, log_sig = h.unflatten(-1, (self.n_layers, 1 + 3 * M)).split((1, M, M, M), dim=-1)
        return a.squeeze(-1), torch.log_softmax(log_w, dim=-1), mu, log_sig.clamp(-7, 3)

    def nll(self, E: Tensor) -> Tensor:
        """Returns the mean negative log-likelihood, over events and layers, of the deposits ``E`` (B, n_layers) [MeV].

        ```
        log p(E_l) = log sigmoid(a_l)                                              E_l = 0
                   = log sigmoid(-a_l) + log sum_m w_lm N(log E_l; mu_lm, sig_lm) - log E_l    E_l > 0
        ```

        ``- log E_l`` makes it the density of E, not of log E. Every event must have a deposit: the decoder takes the
        log of the event's total.
        """
        a, log_w, mu, log_sig = self._head(self.encode(E), E.sum(-1, keepdim=True))
        log_e = torch.log(E.clamp(min=1e-12)).unsqueeze(-1)
        log_n = -0.5 * ((log_e - mu) / torch.exp(log_sig)) ** 2 - log_sig - _HALF_LOG_2PI
        log_p = torch.logsumexp(log_w + log_n, -1) - log_e[..., 0]          # density of E, not of log E
        return -torch.where(E <= 0, F.logsigmoid(a), F.logsigmoid(-a) + log_p).mean()

    @torch.no_grad()
    def sample(self, z: Tensor, tot: Tensor, rng=None) -> Tensor:
        """Samples per-layer deposits that sum to each event's total.

        ```
        k     ~ Categorical(w_l)                         a mixture component per layer l
        log_e = mu_lk + sig_lk n,   n ~ N(0, 1)
        E_l   = 0 with probability sigmoid(a_l),   else exp(min(log_e, log_e_max_l))
        E     = E tot / sum_l E_l                        all zero: the total goes to the layer with the smallest a_l
        ```

        Args:
            z:   (B, K), the standardised latents.
            tot: (B,) or (B, 1), the events' total deposits [MeV]; an event with ``tot <= 0`` gets zeros.
            rng: keys the noise on the event and the layer (``legofmt.rng.EventRNG``); None: torch's noise.

        Returns:
            (B, n_layers), the deposits [MeV].
        """
        z, tot  = z.float(), tot.float().reshape(-1, 1)
        a, log_w, mu, log_sig = self._head(z, tot.clamp(min=1e-6))
        k, n, u = _layer_noise(a, log_w, rng)
        log_e   = _pick(mu, k) + torch.exp(_pick(log_sig, k)) * n
        E       = torch.where(u < torch.sigmoid(a), 0.0, torch.exp(torch.minimum(log_e, self.log_e_max)))
        dead    = E.sum(-1, keepdim=True) == 0      # every layer empty: the total goes to the one least likely to be
        E       = torch.where(dead, E.scatter(-1, a.argmin(-1, keepdim=True), 1.0), E)
        return torch.where(tot > 0, E * tot / E.sum(-1, keepdim=True), 0.0)


def build_edep_ae(spec: dict) -> EdepLayerAE:
    """Constructs the per-layer E_dep autoencoder from ``spec``, frozen.

    Args:
        spec: ``{"config", "state_dict"}``, the output of ``scripts/train_edep_ae.py``, which a flow checkpoint
            keeps in ``model_conf.edep_ae``.

    Returns:
        The autoencoder, frozen and in eval mode.
    """
    ae = EdepLayerAE(**spec["config"])
    ae.load_state_dict(spec["state_dict"])
    ae.requires_grad_(False)
    return ae.eval()


# ---- plumbing ------------------------------------------------------------------------------------------------------

def _pick(p: Tensor, k: Tensor) -> Tensor:
    """Returns ``p`` (B, n_layers, n_mix) at each layer's mixture component ``k`` (B, n_layers, 1): (B, n_layers)."""
    return p.gather(-1, k).squeeze(-1)


def _layer_noise(a: Tensor, log_w: Tensor, rng) -> tuple[Tensor, Tensor, Tensor]:
    """Samples per layer a mixture component (B, n_layers, 1), a standard normal and a uniform (B, n_layers).

    torch's noise, or with ``rng`` keyed on the event and the layer.
    """
    layer = torch.arange(a.shape[-1], device=a.device)
    k = categorical(torch.softmax(log_w, dim=-1), rng, AE_MIX, layer).unsqueeze(-1)
    n = normal_like(a, rng, AE_NORM, layer)
    u = uniform_like(a, rng, AE_ZERO, layer)
    return k, n, u
