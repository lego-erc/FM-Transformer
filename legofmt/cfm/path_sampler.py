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
    v    = sphere_logmap(x0, x1)
    th   = v.norm(dim=-1, keepdim=True)
    c, s = torch.cos(t * th), torch.sin(t * th)
    # normalised like flow_matching's retraction branch, which also maps the E_dep token's (1, 1, 1) direction
    # filler onto the sphere; th = 0 (x_0 = x_1) divides by the smallest float instead
    x_t  = F.normalize(c * x0 + s * v / th.clamp_min(torch.finfo(th.dtype).tiny), dim=-1)
    dx_t = c * v - th * s * x0
    return x_t, dx_t


def _euclidean_geodesic(x0, x1, t):
    x_t  = x0 + t * (x1 - x0)
    dx_t = x1 - x0
    return x_t, dx_t


class ProductPathSampler:
    """The geodesic path on a product manifold, factor by factor."""

    def __init__(self, p_man: ProductManifold):
        self.manifold  = p_man
        self.geodesics = []
        for man in p_man.manifolds:
            if not isinstance(man, (Sphere, Euclidean)):
                raise TypeError(f"no closed-form geodesic for {type(man).__name__}")
            self.geodesics.append(_sphere_geodesic if isinstance(man, Sphere) else _euclidean_geodesic)

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
        t_b   = t.view(-1, *[1] * (bases.dim() - 1))
        dims  = self.manifold.ambient_dims
        parts = [geodesic(x0, x1, t_b)
                 for geodesic, x0, x1 in zip(self.geodesics, bases.split(dims, -1), data.split(dims, -1))]
        x_t, dx_t = (torch.cat(p, -1) for p in zip(*parts))
        return PathSample(x_1=data, x_0=bases, t=t_b.view(-1, 1), x_t=x_t, dx_t=dx_t)
