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
from legofmt.base.base_dists import Angles, Energy
from legofmt.data.struct import _F
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.legacy.gen_base import LegacySampler

_GEOM = GeomTrafos()


def sample_directions(angles, axis, noise, stream=0):
    """Samples unit directions about ``axis``.

    The polar angle to the axis is ``theta = pi w`` with ``w`` drawn from ``angles``; the azimuth ``phi`` is uniform.

    Args:
        angles: the distribution of ``w`` (``Angles``).
        axis:   unit vectors (..., 3), broadcast against the noise: (B, 1, 3) is one axis per event.
        noise:  the draws (``rng.Noise``), one per direction.
        stream: the draw site of keyed noise: ``theta`` reads ``stream``, ``phi`` reads ``stream + 1``.

    Returns:
        Unit vectors ``(*noise.shape, 3)``.
    """
    theta = torch.pi * angles.sample(noise.normal(stream).to(axis))
    phi   = 2 * torch.pi * noise.uniform(stream + 1).to(theta)
    return _GEOM.direction_about(axis, theta, phi)


class GenerateBase(LegacySampler):
    """The sampler of ``x_0`` for a batch.

    Holds the distributions' parameters between ``interface._set_prior`` and the draw (``lego_eval`` reads
    ``edep_mu`` / ``edep_sig`` back).
    """

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

    def __call__(self, ds_t, rng=None):
        """Samples ``x_0`` for a batch.

        Copies the batch and writes a draw on its E_dep token and on every outgoing slot. The conditioning tokens
        (the scalars, the incoming particle) stay the data's.

        Args:
            ds_t: the batch (``DataStruct``).
            rng:  keys the noise on the event and on an outgoing slot's (species, rank) (``legofmt.rng.EventRNG``);
                None: torch's noise.

        Returns:
            ``x_0`` (B, N, 7).
        """
        if not self.is_recipe(): return super().__call__(ds_t, rng)   # any other option
        f      = ds_t.f
        slot   = R.Noise.per_slot(f.out_pid, rng)
        event  = R.Noise(f.edep.shape, f.edep.device, rng)
        energy = Energy(self.sm_scale, self.sm_pow)
        angles = Angles(self.kappa, self.ang_pow)
        in_mom = F.normalize(f.in_mom, dim=-1).unsqueeze(1)   # (B, 1, 3): an event's axis, for all its slots
        in_pos = f.in_pos.unsqueeze(1)

        x0 = _F(f.model_in.clone())
        x0.out_energy[:] = energy.sample(slot.normal(R.BASE_E))
        x0.out_pos[:]    = sample_directions(angles, in_pos, slot, R.BASE_POS)   # torch's noise: position first
        x0.out_mom[:]    = sample_directions(angles, in_mom, slot, R.BASE_MOM)
        x0.edep[:]       = event.normal(R.EDEP)
        return x0.full
