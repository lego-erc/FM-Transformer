import torch
from torch import Tensor

from flow_matching.utils.manifolds import Euclidean, Sphere

from ..geometry.energy_proj import EnergyProjections
from ..geometry.raytracing_proj import CubeTrace
from ..geometry.path_sample_mult import ProductPathSampler, ProductManifold

class DataPrep:
    def __init__(self, config):
        config = config.get("config", config)
        model_conf = config.get("model_conf").copy()
        self.in_dim = model_conf.get("model_args").get("in_dim")
        self.manifold = eval(model_conf.get("manifold"))
        self.proj_ray = model_conf.get("proj_ray", True)
        self.proj_en = model_conf.get("proj_en", False)
        self.proj_en_out = model_conf.get("proj_en_out", False)
        self.pen = EnergyProjections(self.proj_en)
        self.ppa = CubeTrace()

    def __call__(self, batch: tuple) -> Tensor:
        return self.prep(batch)

    @torch.no_grad()
    def prep(self, batch: tuple) -> Tensor:
        cc_ext, mask, attn_mask, data_add = batch
        self.slc = (cc_ext.shape[-1] - self.in_dim) // 2
        cc_ext[..., self.slc:-self.slc] = self.cc_trafo(cc_ext[..., self.slc:-self.slc])
        return self.format_add((cc_ext, mask, attn_mask, data_add))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor) -> Tensor:
        cc = cc.nan_to_num(1)
        if self.proj_en:
            cc = self.pen(cc)
        if self.proj_ray:
            cc[:, 0] = self.ppa(cc[:, 0])
        cc = self.manifold.projx(cc)
        return cc
    
    @torch.no_grad()
    def format_add(self, batch: tuple) -> Tensor:
        cc_ext, mask, attn_mask, data_add = batch
        e_dep = torch.ones_like(cc_ext[:, :1])
        e_in = cc_ext[:, :1, 1:4].norm(dim=-1)
        e_dep[..., self.slc] = data_add.get("E_dep").view_as(e_dep[..., self.slc]) / e_in
        density = torch.ones_like(cc_ext[:, :1])
        density[..., self.slc] = data_add.get("Density").view_as(density[..., self.slc])
        target = torch.cat((density, e_dep, cc_ext), dim=1).nan_to_num()
        target[..., :2, -1] = 0
        mask = torch.cat((torch.zeros_like(mask[:, :1]), 
                          torch.ones_like(mask[:, :1]), mask), dim=1)
        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :2]), attn_mask), dim=1)
        return target, mask, attn_mask