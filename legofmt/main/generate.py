import torch

from .modules import LEGOLtng
from ..geometry.energy_proj import EnergyProjections
from ..multiplicity.gen_mult import GenMult


class GenerateOut(torch.nn.Module):
    def __init__(self, flow_ckpt_path, mult_ckpt_path=None, device="cpu"):
        super().__init__()
        dict_flow = torch.load(flow_ckpt_path, map_location=device, weights_only=False)
        self.model = torch.compile(LEGOLtng(dict_flow).to(device))
        self.en_proj = EnergyProjections()

        dict_mult = torch.load(mult_ckpt_path, map_location=device, weights_only=False)
        config_mult = dict_mult["config"]
        self.gen_mult = GenMult(**config_mult["model_conf"]).to(device)
        self.gen_mult.load_state_dict(dict_mult["state_dict"])

        self.max_particles = config_mult["model_conf"]["max_particles"]

    def __call__(self, incoming_cc, data_add=None):
        masks = self.gen_mult_masks(incoming_cc)
        if data_add is not None:
            batch = (incoming_cc.view(-1, 1, 6).clone(), *masks, data_add)
        else:
            batch = (incoming_cc.view(-1, 1, 6).clone(), *masks)
        sols = self.model(batch)
        return self.en_proj.exp_mult(incoming_cc.view(-1, 1, 6), sols)

    def gen_mult_masks(self, cc):
        mult = self.gen_mult(cc.view(-1, 6))
        attn_mask = ~(
            torch.nn.functional.one_hot(mult, num_classes=self.max_particles)
            .cumsum(-1)
            .bool()
        )
        mask = attn_mask.clone().long().unsqueeze(2)
        mask[:, 0] = 0
        return mask, attn_mask
