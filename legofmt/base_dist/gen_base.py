"""The sampler of the flow's base distribution: a draw of ``x_0``, the state the flow integrates from.

```
energy     u = Energy(s, q).sample(z)                 the outgoing energy, as a fraction of the incoming one
momentum   w = Angles(kappa, pw).sample(z)            its angle to the incoming momentum, / pi; azimuth uniform
position   w = Angles(kappa, pw).sample(z)            the same about the incoming position
E_dep      z                                          the event's deposit, in units of its prior
```

``z ~ N(0, 1)``, one per cell; the distributions are ``base_dists.py``. Their parameters are the config's constants
until the base head sets them for a batch (``interface._set_prior``), per outgoing slot.

This is the recipe's draw (``scale_dist: sm_norm``, ``base_dist: poles``, ``tanh_theta`` with the shape exponent,
``bs_frac: 0``). Every other option is :mod:`legofmt.legacy.gen_base`.
"""

import torch
import torch.nn.functional as F

from legofmt import rng as R
from legofmt.base_dist.base_dists import Angles, Energy
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.legacy.gen_base import LegacySampler

_GEOM = GeomTrafos()


def sample_directions(angles, axis, noise, stream=0):
    """Unit directions ``(*noise.shape, 3)`` about ``axis``: the polar angle ``theta = pi w`` with ``w`` drawn from
    ``angles``, the azimuth ``phi`` uniform. Keyed noise draws them from ``stream`` and ``stream + 1``."""
    theta = torch.pi * angles.sample(noise.normal(stream).to(axis))
    phi   = 2 * torch.pi * noise.uniform(stream + 1).to(theta)
    return _GEOM.direction_about(axis, theta, phi)


class GenerateBase(LegacySampler):
    """Draws ``x_0``: ``__call__`` the outgoing slots, ``insert_add`` the E_dep cell. Holds the distributions'
    parameters between ``interface._set_prior`` and the draw (``lego_eval`` reads ``edep_mu`` / ``edep_sig`` back)."""

    def __init__(self, config: dict):
        base_conf = config.get("base_conf") or {}
        super().__init__(base_conf)   # the options: scale_dist, base_dist, tanh_theta, bs_frac
        self.e_dep_max = base_conf.get("e_dep_max", 1.)
        self.sm_scale  = base_conf.get("sm_scale", 0.5)               # energy s
        self.sm_pow    = 1.0                                          # energy q
        self.kappa     = base_conf.get("kappa", torch.tensor(10.0))   # angles kappa
        self.ang_pow   = None                                         # angles pw: only a base head sets it
        self.edep_mu   = 0.0                                          # E_dep mu
        self.edep_sig  = 1.0                                          # E_dep sig

    def __call__(self, shape, incoming_rt, rng=None, elem=None):
        """``(B, 1 + n_out, 7)``: the incoming token as given, then a draw (energy, momentum, position) for each of
        the ``shape`` = (B, n_out) outgoing slots. Without ``rng`` the noise is torch's; ``rng`` (legofmt.rng.EventRNG)
        keys it per event, ``elem`` (B, n_out) per slot."""
        if not self.is_recipe(): return super().__call__(shape, incoming_rt, rng, elem)   # any other option
        noise  = R.Noise(shape, incoming_rt.device, rng, elem)
        in_mom = F.normalize(incoming_rt[..., 1:4], dim=-1)
        in_pos = incoming_rt[..., -3:]
        angles = Angles(self.kappa, self.ang_pow)
        energy = Energy(self.sm_scale, self.sm_pow).sample(noise.normal(R.BASE_E)[..., None]).to(incoming_rt)
        pos    = sample_directions(angles, in_pos, noise, R.BASE_POS)   # torch's stream: position before momentum
        mom    = sample_directions(angles, in_mom, noise, R.BASE_MOM)
        outgoing = torch.cat((energy, mom, pos), dim=-1)
        return torch.cat((incoming_rt, outgoing), dim=1)

    def insert_add(self, base, rng=None):
        """Fills the E_dep token's energy cell with ``z ~ N(0, 1)``, in place: the flow carries E_dep in units of the
        event's prior (``edep_whiten.py``), where the prior is the standard normal."""
        base[:, 1, 0] = R.Noise(base.shape[:1], base.device, rng).normal(R.EDEP).to(base)
        return base
