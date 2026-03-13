import torch

from ..main.modules import LEGOLtng
from ..multiplicity.model import MultModel


class GenerateOut(torch.nn.Module):
    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu"):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = LEGOLtng(flow_conf).to(device)

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        self.pdgid_in = mult_conf["config"]["mm_conf"]["ptypes_in"]

        self.ntokens = flow_conf["config"]["model_conf"]["model_args"]["ntokens"]

    def __call__(self, cond: torch.Tensor):
        cond = cond.clone()
        masks = self.gen_mult_masks(cond)
        cond_fm = cond[:, None, :]
        cond_fm = torch.cat((torch.zeros_like(cond_fm).expand(-1, 2, -1), cond_fm), dim=1)
        cond_fm[:, 0, 1] = cond[:, 0]
        batch = (cond_fm, *masks)
        return self.model(batch)

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
