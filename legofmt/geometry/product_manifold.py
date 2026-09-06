"""``ProductManifold``: one manifold per slice of the feature dimension.

Each factor's map is applied to its own slice of the last dimension, so the
factor dims must sum to the model's ``in_dim``. The path sampler that trains
against it lives in ``legofmt.cfm.path_sampler``.
"""

import torch
from flow_matching.utils.manifolds import Manifold


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
        return self._batch_map("logmap", x, y)

    def projx(self, x):
        return self._batch_map("projx", x)

    def proju(self, x, u):
        return self._batch_map("proju", x, u)
