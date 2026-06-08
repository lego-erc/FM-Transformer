import torch
import torch.nn.functional as F
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

    def in_frac_log(self, p_x: Tensor) -> Tensor:
        p_x = self.in_frac(p_x)
        p_x[:, 1:] = self.log(p_x[:, 1:])
        return p_x

    def exp_mult(self, in_cc: Tensor, out_cc_frac: Tensor) -> Tensor:
        out_exp = self.exp(out_cc_frac)
        return self.in_mult(in_cc, out_exp)

    def in_frac(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        in_norm = p[:, 0:1].norm(dim=-1, keepdim=True)
        p_normed = torch.cat((p[:, :1], p[:, 1:] / in_norm), dim=1)
        return torch.cat((p_normed, x), -1)

    def in_mult(self, in_cc: Tensor, out_cc_frac: Tensor) -> Tensor:
        p, x = in_cc.split(3, -1)
        in_norm = p.norm(dim=-1, keepdim=True)
        out_p_frac, out_x = out_cc_frac.split(3, -1)
        return torch.cat((out_p_frac * in_norm, out_x), dim=-1)

    def log(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        norm_fac = (1 - p_norm.log()) / p_norm
        return torch.cat((p * norm_fac, x), -1)

    def exp(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True).nan_to_num(0)
        return torch.cat((F.normalize(p, dim=-1) * (1 - p_norm).exp(), x), -1)
