"""The three flow base distributions: for each, how to draw it and its log density.

```
Energy   u = 1 - tanh(s |z|^q)      an outgoing particle's energy, as a fraction of the incoming one
Angles   w = tanh(|z| / kappa)^pw   its momentum / position direction: the angle to the incoming one, / pi
Edep     y = z                      the event's deposited energy, in units of its prior (mu, sig)
```

``z ~ N(0, 1)`` throughout. Each is a class that holds its parameters for a batch and has ``sample(z)`` and
``log_density(x)``; ``Edep`` has no ``sample``, its draw is ``z`` as it is. ``BaseParams`` holds all six parameters:
the base head (``nn.py``) predicts them, ``GenerateBase`` draws with them, ``nn_fit.head_nll`` scores data under
them.

Each class is written the same way: the distribution, its inverse and its log density as plain formulas, then code
that follows them symbol by symbol. The energy and angle distributions are functions of ``|z|``, which is
half-normal:

```
log p(|z|) = log sqrt(2/pi) - |z|^2 / 2
```

so their density in the distribution's own variable ``x`` is ``log p(x) = log p(|z|) + log |d|z| / dx|``.

This file is the formulas and nothing else. Every argument and every result is a tensor, broadcast against the
parameters, and has to lie inside the formula's open domain: ``u`` and ``w`` in (0, 1), ``x`` real. Holding data
there, and everything else a fit does to a density, is ``nn_fit.py``; the parameters' ranges are ``nn.py``.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from legofmt.numerics import safe_pow

_LOG_SQRT_2_OVER_PI = float(torch.log(torch.tensor(2 / torch.pi, dtype=torch.float64))) / 2   # half-normal's constant
_LOG_SQRT_2PI       = float(torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64))) / 2   # normal's constant


class Energy(NamedTuple):
    """An outgoing particle's energy ``u``, as a fraction of the incoming one.

    ```
    sample    u = 1 - tanh(s y),   y = |z|^q

    inverse   y = atanh(1 - u) / s,   |z| = y^(1/q)

    density   log p(u)        = log p(|z|) + log |d|z| / du|
              log p(|z|)      = log sqrt(2/pi) - y^(2/q) / 2
              log |d|z| / du| = (1/q - 1) log y - log(q s u (2 - u))
    ```
    """
    s: Tensor
    q: Tensor

    def sample(self, z):
        s, q = self
        y = torch.abs(z) ** q
        return 1 - torch.tanh(y * s)

    def log_density(self, u):
        s, q    = self
        y       = torch.atanh(1 - u) / s
        log_p_z = _LOG_SQRT_2_OVER_PI - y ** (2/q) / 2
        return (
            log_p_z
            + (1/q - 1) * torch.log(y)           # log |d|z| / du|, its two terms
            - torch.log(q*s*u * (2 - u))
        )


class Angles(NamedTuple):
    """An outgoing particle's angle to the incoming one, ``w = theta / pi`` (momentum and position alike).

    ```
    sample    w = t^pw,   t = tanh(|z| / kappa)

    inverse   t = w^(1/pw),   |z| = kappa atanh(t)

    density   log p(w)        = log p(|z|) + log |d|z| / dw|
              log p(|z|)      = log sqrt(2/pi) - |z|^2 / 2
              log |d|z| / dw| = log kappa - log pw + (1/pw - 1) log w - log(1 - t^2)

    cdf       P(W < w) = P(|Z| < |z|) = erf(|z| / sqrt 2)
    ```
    """
    kappa: Tensor
    pw:    Tensor

    def sample(self, z):
        kappa, pw = self
        t = torch.abs(torch.tanh(z / kappa))
        return safe_pow(t, pw)

    def log_density(self, w):
        kappa, pw = self
        t         = w ** (1/pw)
        abs_z     = kappa * torch.atanh(t)
        log_p_z   = _LOG_SQRT_2_OVER_PI - abs_z**2 / 2
        return (
            log_p_z
            + torch.log(kappa) - torch.log(pw)   # log |d|z| / dw|, its four terms
            + (1/pw - 1) * torch.log(w)
            - torch.log1p(-t**2)
        )

    def cdf(self, w):
        kappa, pw = self
        abs_z = kappa * torch.atanh(w ** (1/pw))
        return torch.erf(abs_z / 2**0.5)


class Edep(NamedTuple):
    """The event's deposited energy ``E``, which the flow carries in units of this prior (``edep_whiten.py``).

    ```
    sample    y = z,   y = (x - mu) / sig,   x = logit(E / e_dep_max)

    density   log p(x) = -log sig - (x - mu)^2 / (2 sig^2) - log sqrt(2 pi)
              (log p(E) adds log |dx/dE|, which is free of mu and sig)
    ```

    The base is ``z`` as it is drawn. ``(mu, sig)`` set the unit; they are fitted so that the data's ``x`` is normal.
    """
    mu:  Tensor
    sig: Tensor

    def log_density(self, x):
        mu, sig = self
        return -(torch.log(sig) + (x - mu) ** 2 / (2 * sig**2) + _LOG_SQRT_2PI)


class BaseParams(NamedTuple):
    """The three distributions' parameters for a batch. One column per outgoing class ``(B, n_cls)`` as the head
    returns them, or gathered onto the outgoing slots ``(B, n_out)``; E_dep's are per event ``(B, 1)``."""
    sm_scale: Tensor   # energy s
    edep_mu:  Tensor   # E_dep mu
    edep_sig: Tensor   # E_dep sig
    kappa:    Tensor   # angles kappa
    ang_pow:  Tensor   # angles pw
    sm_pow:   Tensor   # energy q

    @property
    def energy(self):
        return Energy(self.sm_scale, self.sm_pow)

    @property
    def angles(self):
        return Angles(self.kappa, self.ang_pow)

    @property
    def edep(self):
        return Edep(self.edep_mu, self.edep_sig)
