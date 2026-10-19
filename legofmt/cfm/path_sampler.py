"""The probability path the flow trains against.

``ProductPathSampler`` samples the geodesic ``x_t = exp_{x_0}(t log_{x_0} x_1)``
(flow_matching's ``GeodesicProbPath`` with the CondOT schedule) and its velocity
per manifold factor, in closed form: the vmap + forward-mode jvp it replaces
cost ~4 ms of a ~80 ms training step.
"""

import torch
import torch.nn.functional as F
from flow_matching.path.path_sample import PathSample
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.geometry.product_manifold import ProductManifold, sphere_logmap


class ProductPathSampler:
    def __init__(self, p_man: ProductManifold):
        self.manifold = p_man

    def sample(self, bases, data, t):
        """``t`` is one time per event, broadcast over the sequence; the returned
        ``t`` is per token, shaped like ``bases`` without its last dimension."""
        t_b = t.view(-1, *[1] * (bases.dim() - 1))
        x_t, dx_t = [], []
        for man, x0, x1 in zip(self.manifold.manifolds,
                                bases.split(self.manifold.ambient_dims, -1),
                                data.split(self.manifold.ambient_dims, -1)):
            if isinstance(man, Sphere):
                v  = sphere_logmap(x0, x1)
                th = v.norm(dim=-1, keepdim=True)
                c, s = (t_b * th).cos(), (t_b * th).sin()
                # normalised like flow_matching's retraction branch, which also maps
                # the E_dep row's (1, 1, 1) direction filler onto the sphere
                x_t.append(F.normalize(c * x0 + s * v / th.clamp_min(torch.finfo(th.dtype).tiny), dim=-1))
                dx_t.append(c * v - th * s * x0)
            elif isinstance(man, Euclidean):
                x_t.append(x0 + t_b * (x1 - x0))
                dx_t.append(x1 - x0)
            else:
                raise TypeError(f"no closed-form geodesic for {type(man).__name__}")
        return PathSample(x_1=data, x_0=bases, t=t_b[..., 0].expand(bases.shape[:-1]).contiguous(),
                          x_t=torch.cat(x_t, -1), dx_t=torch.cat(dx_t, -1))
