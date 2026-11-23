"""The state that a solve and the flow-map term move: the tokens on the manifold, and the event's latent beside them.

```
state     s = (x, z)                 x: the tokens (B, L, 7) on the product manifold, z: the edep_ae latent (B, K)

vectors   s + u,  s - u,  c s,  s / c   componentwise; c a scalar, or a State of one factor per part

maps      exp_s(u)  = (exp_x(u.x), z + u.z)
          log_s(e)  = (log_x(e.x), e.z - z)
          proj_s(u) = (proj_x(u.x), u.z)          onto the tangent space at s
          proj_M(s) = (proj_M(x), z)              onto the manifold
```

``z`` is Euclidean: the manifold's maps act on ``x`` alone. A model without a latent carries an empty one
(``K = 0``, ``no_latent``), so every step is written once, for models with and without a latent. ``state_field`` is
the field on a state; ``moved`` applies a step to the generated tokens and the latent only.
"""

from typing import NamedTuple

import torch
from torch import Tensor


class State(NamedTuple):
    """The tokens ``x`` (B, L, 7) and the latent ``z`` (B, K); a state, or a velocity at one.

    Adds, subtracts and scales componentwise. A ``State`` as the factor scales each part by its own entry: one step
    size per event is ``State((B, 1, 1), (B, 1))``.
    """
    x: Tensor
    z: Tensor

    def __add__(a, b):
        return State(a.x + b.x, a.z + b.z)

    def __sub__(a, b):
        return State(a.x - b.x, a.z - b.z)

    def __rmul__(a, c):
        return State(c * a.x, c * a.z)

    def __mul__(c, a):
        return State(c.x * a.x, c.z * a.z)

    def __truediv__(a, c):
        c_x, c_z = c if isinstance(c, State) else (c, c)
        return State(a.x / c_x, a.z / c_z)


class LatentManifold(NamedTuple):
    """The product manifold's maps on a ``State``: they act on ``x``, ``z`` is Euclidean."""
    man: object

    def expmap(self, s, u):
        return State(self.man.expmap(s.x, u.x), s.z + u.z)

    def logmap(self, s, e):
        return State(self.man.logmap(s.x, e.x), e.z - s.z)

    def proju(self, s, u):
        return State(self.man.proju(s.x, u.x), u.z)

    def projx(self, s):
        return State(self.man.projx(s.x), s.z)


def state_field(model, extras: dict, clone: bool = False):
    """Returns the field on a ``State``: ``model`` with the batch's keyword inputs ``extras``.

    With an empty latent ``model`` is called without ``z``, and the empty latent is returned as its own velocity.

    Args:
        model:  the field, called as ``model(x, t, z=..., **d, **extras)``.
        extras: the batch's keyword inputs of ``model``: ``mask``, ``attn_mask``, ``types``, ``pdgids``.
        clone:  copy every velocity. A field compiled with CUDA graphs overwrites its output on the next call, and
            rk4 holds its stages across calls.

    Returns:
        ``field(s, t, **d) -> State``: the velocity at the state ``s`` and time ``t``; ``d=...`` is the step size a
        flow-map field is told.
    """
    def field(s, t, **d):
        if s.z.shape[-1] == 0:
            v = (model(s.x, t, **d, **extras), s.z)
        else:
            v = model(s.x, t, z=s.z, **d, **extras)
        return State(*(a.clone() for a in v)) if clone else State(*v)
    return field


def no_latent(x: Tensor) -> Tensor:
    """Returns the empty latent (B, 0) that a model without a latent carries through its steps."""
    return x.new_zeros(len(x), 0)


def moved(gen: Tensor, s: State, s_new: State) -> State:
    """Returns ``s_new`` on the generated tokens ``gen`` (B, L, 1) and the latent, ``s`` on every other token."""
    return State(torch.where(gen, s_new.x, s.x), s_new.z)
