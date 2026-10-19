"""``ProductManifold``: one manifold per slice of the feature dimension.

Each factor's map is applied to its own slice of the last dimension, so the
factor dims must sum to the model's ``in_dim``. The path sampler that trains
against it lives in ``legofmt.cfm.path_sampler``.
"""

import torch
from flow_matching.utils.manifolds import Manifold, Sphere


def sphere_logmap(x, y):
    """``Sphere.logmap`` with the angle from ``atan2(|u|, x.y)``: flow_matching's
    ``acos(x.y)`` keeps few digits of small angles in fp32 (median relative error
    2% at 1e-3 rad, 15% at 3e-4), the scale of one short flow-map step."""
    u  = (y - x) - (x * (y - x)).sum(-1, keepdim=True) * x
    su = u.norm(dim=-1, keepdim=True)
    th = torch.atan2(su, (x * y).sum(-1, keepdim=True))
    return u * (th / su.clamp_min(torch.finfo(su.dtype).tiny))


class ProductManifold(Manifold):
    def __init__(self, manifolds: list, ambient_dims: tuple):
        if len(manifolds) != len(ambient_dims):
            raise ValueError("Number of Manifolds must match ambient_dims length!")

        super().__init__()
        self.manifolds = manifolds
        self.ambient_dims = ambient_dims

    def _batch_map(self, fn_name, *tensors, **kwargs):
        parts = [t.split(self.ambient_dims, dim=-1) for t in tensors]
        results = [
            getattr(man, fn_name)(*[p[i] for p in parts], **kwargs)
            for i, man in enumerate(self.manifolds)
        ]
        return torch.cat(results, dim=-1)

    def expmap(self, x, u):
        return self._batch_map("expmap", x, u)

    def logmap(self, x, y):
        return torch.cat([
            sphere_logmap(a, b) if isinstance(man, Sphere) else man.logmap(a, b)
            for man, a, b in zip(self.manifolds, x.split(self.ambient_dims, -1), y.split(self.ambient_dims, -1))
        ], dim=-1)

    def projx(self, x):
        return self._batch_map("projx", x)

    def proju(self, x, u):
        return self._batch_map("proju", x, u)
