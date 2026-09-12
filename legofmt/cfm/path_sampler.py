"""The probability path the flow trains against.

``ProductPathSampler`` runs one ``GeodesicProbPath`` per manifold factor over
that factor's slice of the last dimension, then stitches the per-factor samples
back into a single ``(x_t, dx_t, t)`` triple shaped like the input batch.
"""

import torch
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler, Scheduler

from legofmt.geometry.product_manifold import ProductManifold


class ProductPath:
    def __init__(self, paths, view_as_: tuple):
        self.x_t = torch.cat([p.x_t for p in paths], dim=-1).view(*view_as_)
        self.dx_t = torch.cat([p.dx_t for p in paths], dim=-1).view(*view_as_)
        self.t = paths[0].t.view(view_as_[:-1])


class ProductPathSampler:
    def __init__(
        self, p_man: ProductManifold, scheduler: Scheduler = CondOTScheduler()
    ):
        self.ambient_dims = p_man.ambient_dims
        self.paths = [
            GeodesicProbPath(scheduler=scheduler, manifold=manifold)
            for manifold in p_man.manifolds
        ]

    def sample(self, bases, data, t):
        bases_ = bases.split(self.ambient_dims, dim=-1)
        data_ = data.split(self.ambient_dims, dim=-1)
        t = t.repeat_interleave(bases.shape[1:-1].numel())
        paths = [
            path.sample(bases_[i].flatten(0, -2), data_[i].flatten(0, -2), t)
            for i, path in enumerate(self.paths)
        ]
        return ProductPath(paths, bases.shape)
