import torch
from torch import Tensor

class EnergyProjections:
    def __init__(self, norm_type: str | bool = "in_frac"):
        self.func = self.__getattribute__(norm_type)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

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
        p_normed = torch.cat((p[:, 0:1], p[:, 1:] / in_norm), dim=1)
        return torch.cat((p_normed, x), -1)

    def in_mult(self, in_cc: Tensor, out_cc_frac: Tensor) -> Tensor:
        p, x = in_cc.split(3, -1)
        in_norm = p.norm(dim=-1, keepdim=True)
        out_p_frac, out_x = out_cc_frac.split(3, -1)
        return torch.cat((out_p_frac * in_norm, out_x), dim=-1)

    def log(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True)
        norm_fac = p_norm.log() / p_norm
        return torch.cat((p * norm_fac.abs(), x), -1)

    def exp(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True).nan_to_num(0)
        return torch.cat((p / p_norm * (-p_norm.abs()).exp(), x), -1)

    def factor(self, p_x: Tensor) -> Tensor:
        return p_x / torch.log(torch.tensor([10 / 300], device=p_x.device)).abs()
