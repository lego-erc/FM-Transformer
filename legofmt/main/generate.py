import torch

import sys

from ..main.modules import LEGOLtng
from ..multiplicity.model import MultModel
from ..geometry.energy_proj import EnergyProjections
from ..geometry.raytracing_proj import CubeTrace


class GenerateOut(torch.nn.Module):
    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu"):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = LEGOLtng(flow_conf).to(device)

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        self.pdgid_in = mult_conf["config"]["mm_conf"]["ptypes_in"]

        self.ntokens = flow_conf["config"]["model_conf"]["model_args"]["ntokens"]
        self.proj_en = flow_conf["config"]["model_conf"]["proj_en"]
        self.pen = EnergyProjections(self.proj_en)
        self.proj_ray = CubeTrace()

    def __call__(self, cond: torch.Tensor, gen_gt: bool = False):
        cond_model = cond.clone()
        if self.proj_en is not False:
            cond_model[..., 1:7] = self.pen(cond_model[..., 1:7])
        cond_model[..., 1:7] = self.proj_ray(cond_model[..., 1:7])
        masks = self.gen_mult_masks(cond_model)
        cond_fm = cond_model[:, None, :]
        cond_fm = torch.cat(
            (torch.zeros_like(cond_fm).expand(-1, 2, -1), cond_fm), dim=1
        )
        cond_fm[:, 0, 1] = cond_model[:, 0]
        batch = (cond_fm, *masks)
        model_out = self.model(batch)
        if not gen_gt:
            return model_out
        import pyg4lego
        cond_gt = cond.clone().cpu()
        mom = cond_gt[0, 1:4].double()
        pos = cond_gt[0, 4:7].double()
        energy = mom.norm(dim=-1, keepdim=True).double()
        density = cond_gt[0, :1].double()
        size = torch.tensor([100.0], dtype=torch.float64)
        pdgids = cond_gt[0, -1:].int()

        gt_out = pyg4lego.run_simulation(
            cond_gt.shape[0],
            pos,
            mom,
            energy,
            random_gun=False,
            density=density,
            size=size,
            random_energy=False,
            SourceParticles=pdgids,
            savepath=None,
        )

        return {
            "model_out": model_out,
            "gt_out": gt_out,
        }

    def gen_mult_masks(self, cond: torch.Tensor):
        pdgid_in = cond[:, -1].long()
        pdgid_in_idx = torch.searchsorted(self.pdgid_in, pdgid_in)
        mult = self.gen_mult((cond[:, 0:-1], None, pdgid_in_idx))

        idx = torch.arange(self.ntokens - 3, device=mult.device)
        attn_mask = idx < mult.sum(-1, keepdim=True)

        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :3]), attn_mask), dim=1)
        mask = attn_mask.clone().long().unsqueeze(2)
        mask[:, [0, 2]] = 0
        return mask, attn_mask
