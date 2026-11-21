"""The probability path the flow trains against: the geodesic from the base sample ``x_0`` to the data ``x_1``.

```
x_t = exp_{x_0}(t log_{x_0}(x_1))             flow_matching's GeodesicProbPath with the CondOT schedule

sphere      v    = log_{x_0}(x_1),   th = |v|
            x_t  = cos(t th) x_0 + sin(t th) v / th
            dx_t = cos(t th) v - th sin(t th) x_0

Euclidean   x_t  = x_0 + t (x_1 - x_0)
            dx_t = x_1 - x_0
```

``dx_t`` is the velocity the field regresses on. Both are written out per factor of the product manifold: the vmap +
forward-mode jvp this replaces cost ~4 ms of a ~80 ms training step.
"""

import torch
import torch.nn.functional as F
from flow_matching.path.path_sample import PathSample
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.geometry.product_manifold import ProductManifold, sphere_logmap


def _sphere_geodesic(x0, x1, t):
    """Returns the point ``x_t`` and the velocity ``dx_t`` at ``t`` of the great circle from ``x0`` to ``x1``.

    ``x_t`` is normalised like flow_matching's retraction branch, which also maps the E_dep token's (1, 1, 1) direction
    filler onto the sphere; ``th = 0`` (x_0 = x_1) divides by the smallest float instead.
    """
    v    = sphere_logmap(x0, x1)
    th   = v.norm(dim=-1, keepdim=True)
    phi  = t * th
    c, s = torch.cos(phi), torch.sin(phi)
    x_t  = F.normalize(
        c * x0
        + s * v / th.clamp(min=torch.finfo(th.dtype).tiny),
        dim=-1,
    )
    dx_t = c * v - th * s * x0
    return x_t, dx_t


def _euclidean_geodesic(x0, x1, t):
    """Returns the point ``x_t`` and the velocity ``dx_t`` at ``t`` of the straight line from ``x0`` to ``x1``."""
    dx_t = x1 - x0
    return x0 + t * dx_t, dx_t


class ProductPathSampler:
    """The geodesic path on a product manifold, factor by factor."""

    def __init__(self, p_man: ProductManifold):
        self.manifold  = p_man
        self.geodesics = [_geodesic_of(man) for man in p_man.manifolds]

    def sample(self, bases, data, t):
        """Evaluates the geodesic from ``bases`` to ``data`` at time ``t``: the point ``x_t`` and its velocity ``dx_t``.

        Args:
            bases: (B, L, 7), ``x_0``.
            data:  (B, L, 7), ``x_1``.
            t:     (B,), one time per event, broadcast over the sequence.

        Returns:
            ``PathSample`` with ``x_0``, ``x_1``, ``x_t``, ``dx_t`` and ``t`` as (B, 1), the per-event form the
            vector field conditions on.
        """
        t_b       = _per_event(t, bases)                    # (B, 1, 1): over the tokens and channels
        x_t, dx_t = _per_factor(self.geodesics, self.manifold.ambient_dims, bases, data, t_b)
        return PathSample(x_1=data, x_0=bases, t=t_b.flatten(1), x_t=x_t, dx_t=dx_t)


# ---- plumbing ------------------------------------------------------------------------------------------------------

def _geodesic_of(man):
    """Returns the closed-form geodesic of a ``Sphere`` or ``Euclidean`` factor.

    Raises:
        TypeError: any other manifold.
    """
    if not isinstance(man, (Sphere, Euclidean)):
        raise TypeError(f"no closed-form geodesic for {type(man).__name__}")
    return _sphere_geodesic if isinstance(man, Sphere) else _euclidean_geodesic


def _per_event(t, like):
    """Shapes one time per event (B,) to broadcast over the other dims of ``like``."""
    return t.reshape(-1, *[1] * (like.dim() - 1))


def _per_factor(geodesics, dims, x0, x1, t):
    """Applies each factor's geodesic to its own ``dims`` channels; returns ``x_t`` and ``dx_t``, joined back."""
    parts     = [geodesic(a, b, t) for geodesic, a, b in zip(geodesics, x0.split(dims, dim=-1), x1.split(dims, dim=-1))]
    x_t, dx_t = (torch.cat(p, dim=-1) for p in zip(*parts))
    return x_t, dx_t
