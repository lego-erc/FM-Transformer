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


def sphere_expmap(x, u):
    """``Sphere.expmap`` with its arithmetic (bit-identical), differentiable at ``u = 0`` (every conditioning row of a
    solve): the division sees 1 where the velocity is below EPS, so the unused branch of the ``where`` carries no
    ``0/0`` (``norm``'s backward is 0 at a zero norm)."""
    n = u.norm(dim=-1, keepdim=True)
    big = n > Sphere.EPS[n.dtype]
    nu = torch.where(big, n, torch.ones_like(n))
    retr = x + u
    return torch.where(big, x * torch.cos(nu) + u * torch.sin(nu) / nu, retr / retr.norm(dim=-1, keepdim=True))


_SPHERE_MAPS = {"expmap": sphere_expmap, "logmap": sphere_logmap}   # used in place of flow_matching's Sphere methods


class ProductManifold(Manifold):
    def __init__(self, manifolds: list, ambient_dims: tuple):
        if len(manifolds) != len(ambient_dims):
            raise ValueError("Number of Manifolds must match ambient_dims length!")

        super().__init__()
        self.manifolds = manifolds
        self.ambient_dims = ambient_dims

    def _batch_map(self, fn_name, *tensors):
        fns = [_SPHERE_MAPS[fn_name] if isinstance(man, Sphere) and fn_name in _SPHERE_MAPS else getattr(man, fn_name)
               for man in self.manifolds]
        parts = zip(*(t.split(self.ambient_dims, dim=-1) for t in tensors))
        return torch.cat([fn(*p) for fn, p in zip(fns, parts)], dim=-1)

    def expmap(self, x, u):
        return self._batch_map("expmap", x, u)

    def logmap(self, x, y):
        return self._batch_map("logmap", x, y)

    def projx(self, x):
        return self._batch_map("projx", x)

    def proju(self, x, u):
        return self._batch_map("proju", x, u)
