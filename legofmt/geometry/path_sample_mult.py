import torch
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler, Scheduler
from flow_matching.utils.manifolds import Manifold


class ProductManifold(Manifold):
    def __init__(self, manifolds: list, ambient_dims: tuple):
        if len(manifolds) != len(ambient_dims):
            raise ValueError("Number of Manifolds must match ambient_dims length!")

        super().__init__()
        self.manifolds = manifolds
        self.ambient_dims = ambient_dims

    def _batch_map(self, fn_name, *tensors, **kwargs):
        vars = [t.split(self.ambient_dims, dim=-1) for t in tensors]
        results = [
            getattr(man, fn_name)(*[split[i] for split in vars], **kwargs)
            for i, man in enumerate(self.manifolds)
        ]
        return torch.cat(results, dim=-1)

    def expmap(self, x, u):
        return self._batch_map("expmap", x, u)

    def logmap(self, x, y):
        return self._batch_map("logmap", x, y)

    def projx(self, x):
        return self._batch_map("projx", x)

    def proju(self, x, u):
        return self._batch_map("proju", x, u)

    def dist(self, x, y, keepdim=False):
        return self._batch_map("dist", x, y, keepdim=keepdim)


class ProductPath:
    def __init__(self, paths, view_as_: tuple):
        self.x_t = torch.cat([p.x_t for p in paths], dim=-1).view(*view_as_)
        self.dx_t = torch.cat([p.dx_t for p in paths], dim=-1).view(*view_as_)
        self.t = paths[0].t.view(*view_as_[:-1])


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
