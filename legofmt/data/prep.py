import torch
import torch.nn.functional as F
from torch import Tensor

from ..geometry.energy_proj import EnergyProjections
from ..geometry.raytracing_proj import CubeTrace
from ..mod_comps.config import build_manifold
from .struct import _F

class DataPrep:
    def __init__(self, config):
        config = config.get("config", config)
        if "model_conf" in config:
            model_conf = config.get("model_conf").copy()
            self.manifold = build_manifold(model_conf.get("manifold"))
            self.proj_ray = model_conf.get("proj_ray", True)
            cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev")
            max_energy = model_conf["max_energy"]
        else:
            self.manifold = build_manifold(config.get("manifold"))
            self.proj_ray = config.get("proj_ray")
            cutoff_mev = config.get("cutoff_mev")
            max_energy = config.get("max_energy")
        self.pen = EnergyProjections(
            cutoff_mev=cutoff_mev,
            max_energy=max_energy,
        )
        self.ppa = CubeTrace()

    def __call__(self, batch: tuple) -> Tensor:
        return self.prep(batch)

    @torch.no_grad()
    def prep(self, batch: tuple) -> Tensor:
        cc_ext, mask, attn_mask, data_add = batch
        model_in = self.cc_trafo(cc_ext[..., 1:7])
        cc_ext = torch.cat((model_in, cc_ext[..., 7:]), dim=-1)
        return self.format_add((cc_ext, mask, attn_mask, data_add))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor) -> Tensor:
        cc = cc.nan_to_num(1)
        mom, pos = cc.split(3, -1)
        dir_, e = self.pen.to_scalar(mom)
        e = torch.cat((e[:, :1], 1 - (e[:, 1:] / e[:, :1].clamp_min(1e-6)).clamp(0, 1)), dim=1)
        if self.proj_ray:
            ray = torch.cat((dir_[:, 0], pos[:, 0]), dim=-1)
            pos = pos.clone()
            pos[:, 0] = self.ppa(ray)[..., 3:]
        return self.manifold.projx(
            torch.cat((e, F.normalize(dir_, dim=-1), F.normalize(pos, dim=-1)), dim=-1)
        )

    @torch.no_grad()
    def format_add(self, batch: tuple) -> Tensor:
        cc_ext, mask, attn_mask, data_add = batch
        e_dep = torch.ones_like(cc_ext[:, :1])
        e_dep[..., 0] = data_add.get("E_dep").view_as(e_dep[..., 0]) / self.pen.max_energy
        density = torch.ones_like(cc_ext[:, :1])
        density[..., 0] = data_add.get("Density").view_as(density[..., 0])
        target = torch.cat((density, e_dep, cc_ext), dim=1).nan_to_num()
        _F(target).non_p[..., -1] = 0
        mask = torch.cat((torch.zeros_like(mask[:, :1]),
                          torch.ones_like(mask[:, :1]), mask), dim=1)
        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :2]), attn_mask), dim=1)
        return target, mask, attn_mask
