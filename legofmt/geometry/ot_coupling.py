import torch
from torch_linear_assignment import batch_linear_assignment as bla

class OTCoupling:

    def __init__(self, sbset_s = 2**8, method='mean_assignment', ma_pow=2):
        self.sbsset_s = sbset_s
        self.func = self.__getattribute__(method)
        self.ma_pow = ma_pow

    def __call__(self, base, target, attn_mask):
        real_sel = attn_mask[:2**12].flatten()
        cost, base_pad, nt_real = self.comp_cost(base, target, real_sel)
        assign = self.func(cost)
        base_assn = base_pad.gather(1, assign.unsqueeze(-1).expand_as(base_pad))
        base.flatten(0,1)[real_sel] = base_assn.flatten(0,1)[:nt_real]
        return base

    def comp_cost(self, base, target, real_sel):
        dc_real = target.flatten(0,1)[real_sel]
        base_real = base.flatten(0,1)[real_sel]
        nt_real = torch.tensor([dc_real.shape[0]], device=base.device)
        pad_len = self.sbsset_s - torch.remainder(nt_real - 1, self.sbsset_s) - 1
        c_shape = base.shape[-1]
        z_init = torch.zeros((pad_len, c_shape), device=base.device)
        dc_pad = torch.cat((dc_real, z_init), dim=0).view(-1, self.sbsset_s, c_shape)
        base_pad = torch.cat((base_real, z_init), dim=0).view(-1, self.sbsset_s, c_shape)
        return torch.cdist(dc_pad, base_pad), base_pad, nt_real

    def mean_assignment(self, cost):
        cost = cost**self.ma_pow
        mean = cost.mean(dim=(-2, -1), keepdim=True)
        c_max = cost.amax(dim=(-2, -1), keepdim=True)
        zr = torch.full_like(cost, torch.inf, device=cost.device)
        substr = 0

        while True:
            for _ in range(15):
                cost_red = (cost - mean).abs() - substr
                min_obj, min_obj_idx = cost_red.min(dim=-1, keepdim=True)
                min_bid, min_bidder = zr.scatter(-1, min_obj_idx, min_obj).min(dim=-2, keepdim=True)
                zr = zr.scatter(-2, min_bidder, min_bid)
                has_bid = zr.isfinite().any(dim=-2, keepdim=True)
                has_asset = zr.isfinite().any(dim=-1, keepdim=True)
                substr = (~has_asset * ~has_bid).int() * c_max
            if has_bid.all():
                break
        
        return zr.argmin(-1)
    
    def tla_bla(self, cost):
        return bla(cost)