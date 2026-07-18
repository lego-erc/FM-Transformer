import torch
from torch import Tensor

class EnergyProjections:
    def __init__(self, norm_type: str | bool = "in_frac", cutoff_mev: float = 10.0, max_energy: float | None = None):
        self.func = getattr(self, norm_type) if isinstance(norm_type, str) else self.identity
        self.cutoff = cutoff_mev
        self.max_energy = max_energy
        self.log_range = torch.tensor(max_energy / cutoff_mev).log().item() if max_energy else None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def identity(self, p_x: Tensor) -> Tensor:
        return p_x

    def to_scalar(self, mom: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        norm = mom.norm(dim=-1, keepdim=True)
        e = (torch.log(norm.clamp_min(eps) / self.cutoff) / self.log_range).clamp(0.0, 1.0)
        return mom / norm.clamp_min(eps), e

    def from_scalar(self, dir_: Tensor, e: Tensor) -> Tensor:
        norm = self.cutoff * (self.max_energy / self.cutoff) ** e.clamp(0.0, 1.0)
        return dir_ * norm

    def to_mev(self, e_model: Tensor, e_in: Tensor) -> Tensor:
        s_out = (1 - e_model.clamp(0.0, 1.0)) * e_in.clamp(0.0, 1.0)
        return self.cutoff * (self.max_energy / self.cutoff) ** s_out

    def in_frac_log(self, p_x: Tensor) -> Tensor:
        p_x = self.in_frac(p_x)
        p_x[:, 1:] = self.log(p_x[:, 1:])
        return p_x

    def in_frac(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        in_norm = p[:, 0:1].norm(dim=-1, keepdim=True)
        p_normed = torch.cat((p[:, :1], p[:, 1:] / in_norm), dim=1)
        return torch.cat((p_normed, x), -1)

    def log(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat((p * (1 - p_norm.log()) / p_norm, x), -1)
