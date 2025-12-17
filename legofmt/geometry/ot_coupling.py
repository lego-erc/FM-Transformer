import torch
from torch_lap_cuda_lib import solve_lap as slap

class OTCoupling:

    def __init__(self):
        pass

    def __call__(self, base, target, attn_mask):
        cost = self.comp_cost(base, target, attn_mask)
        assign = slap(cost, cost.device)
        return base.gather(1, assign.unsqueeze(-1).expand_as(base))

    def comp_cost(self, base, target, attn_mask):
        return torch.cdist(target, base) 