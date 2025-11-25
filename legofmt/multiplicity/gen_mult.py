import torch
from flow_matching.utils.manifolds import Euclidean, Sphere
from torch import nn

from ..geometry.path_sample_mult import ProductManifold


class MultCondsMLP(nn.Module):
    def __init__(self, max_particles, h_dim=2**8):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(6, h_dim),
            nn.Mish(),
            nn.Dropout(0.02),
            nn.Linear(h_dim, h_dim),
            nn.Mish(),
            nn.Dropout(0.02),
            nn.Linear(h_dim, h_dim),
            nn.Mish(),
            nn.Dropout(0.02),
            nn.Linear(h_dim, h_dim),
            nn.Mish(),
            nn.Dropout(0.02),
            nn.Linear(h_dim, h_dim),
            nn.Mish(),
            nn.Linear(h_dim, max_particles),
        )

    def forward(self, p_x):
        return self.func(p_x)


class GenMult(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        weight = kwargs.pop("weight", None)
        self.model = MultCondsMLP(*args, **kwargs)
        self.man_euc_sph = ProductManifold([Euclidean(), Sphere()], (3, 3))
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def loss(self, cc, n_target):
        logits = self.model(self.man_euc_sph.projx(cc))
        return self.ce_loss(
            logits, n_target
        )

    def generate(self, cc, logits_only=False):
        logits = self.model(self.man_euc_sph.projx(cc))
        if logits_only:
            return logits
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()
