import torch

from ..main.modules import LEGOLtng
from ..geometry.energy_proj import EnergyProjections
from ..multiplicity.model import MultModel


class GenerateOut(torch.nn.Module):
    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu"):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = LEGOLtng(flow_conf).to(device)
        self.en_proj = EnergyProjections()

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        self.ntokens = flow_conf["config"]["model_conf"]["ntokens"]

    def __call__(self, cond: torch.Tensor):
        masks = self.gen_mult_masks(cond[:, -1:])
        batch = (cond.clone(), *masks)
        return self.model(batch)

    def gen_mult_masks(self, cond: torch.Tensor):
        if cond.ndim == 3 and cond.shape[1] == 2:
            density, cc = cond.split(1, 1)
        else:
            cc = cond
        pdgid_in = cond[:, 0, -1].long()
        pdgid_in_idx = (pdgid_in == pdgid_in.unique().view(-1, 1)).nonzero()[:, 0]
        mult = self.gen_mult((cc[:, 0, 1:-1], None, pdgid_in_idx))
        attn_mask = ~(
            torch.nn.functional.one_hot(mult.sum(-1).clamp(max=self.ntokens-4), num_classes=self.ntokens-3)
            .cumsum(-1)
            .bool()
        )
        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :3]), attn_mask), dim=1)
        mask = attn_mask.clone().long().unsqueeze(2)
        mask[:, [0, 2]] = 0
        return mask, attn_mask
