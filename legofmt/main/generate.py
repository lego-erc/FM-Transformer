import torch

from ..main.modules import LEGOLtng
from ..multiplicity.model import MultModel
from ..geometry.energy_proj import EnergyProjections
from ..geometry.raytracing_proj import CubeTrace


class GenerateOut(torch.nn.Module):
    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu", couple_in_out_pdgids=False):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = LEGOLtng(flow_conf).to(device)
        self.model.pdgid_is_idx = True

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        self.pdgid_in = mult_conf["config"]["mm_conf"]["ptypes_in"].to(device)
        self.ptypes = mult_conf["config"]["mm_conf"]["ptypes"].to(device)

        if couple_in_out_pdgids:
            self.model.odeint_conf["filter_pdgid"] = self.pdgid_in

        self.max_seq_l = flow_conf["config"]["model_conf"]["model_args"]["max_seq_l"]
        self.pdgids = flow_conf["config"]["model_conf"]["pdgids"].to(device)
        self.valid_ptypes_mask = torch.isin(self.ptypes, self.pdgids)
        self.proj_ray = CubeTrace()

    def __call__(self, cond: torch.Tensor, prepped: bool = False):
        model_out = self.proj_ray_pass_to_model(cond, prepped=prepped)
        return model_out

    def proj_ray_pass_to_model(self, cond: torch.Tensor, prepped: bool = False):
        cond_model = cond.clone()
        if not prepped:
            cond_model[..., 1:7] = self.proj_ray(cond_model[..., 1:7])
        batch = self.gen_batch(cond_model)
        sols, mask, attn_mask = self.model(batch)
        sols[..., -1] = torch.cat([sols.new_zeros(1), self.pdgids.to(sols.dtype)])[sols[..., -1].long()]
        return sols, mask, attn_mask
    
    def gen_model_w_g4_args(self, n, pos, mom, energy, density, size, pdgids):
        mom_s = mom.view(-1, 3).shape[0]
        pos_s = pos.view(-1, 3).shape[0]
        energy_s = energy.shape[0]
        density_s = density.shape[0]

        if not size.shape[0] == 1:
            raise ValueError("Multiple sizes not yet supported.")
        if not mom_s == pos_s:
            raise ValueError("Mismatching mom and pos batch size.")
        if density_s > 1 and not energy_s > 1:
            raise ValueError("Only one of energy and density can have batch size > 1.")
        if pos_s > 1 and (energy_s > 1 or density_s > 1):
            raise ValueError("If different coordinates are given, energy and density must have batch size 1.")

        cc = torch.cat((mom, pos), dim=-1).view(-1, 6)

        if pos_s > 1:
            cc = cc.repeat_interleave(n, dim=0)
            d = density.view(-1, 1).expand_as(cc[:, :1])
            cc[..., :3] = cc[..., :3] * energy.view(1, 1)

        elif energy_s > 1:
            e = energy.repeat_interleave(n, dim=0).view(-1, 1)
            d = density.view(-1, 1).expand_as(e)
            cc = torch.cat((cc[..., :3].view(1, 3) * e, cc[..., 3:].view(1, 3).expand(e.shape[0], -1)), dim=-1)

        else:
            d = density.repeat_interleave(n, dim=0).view(-1, 1)
            e = energy.view(-1, 1).expand_as(d)
            cc = torch.cat((cc[..., :3].view(1, 3) * e, cc[..., 3:].view(1, 3).expand(e.shape[0], -1)), dim=-1)

        ptypes = pdgids[torch.randint(pdgids.shape[0], d.shape, device=d.device)]
        cond = torch.cat((d, cc, ptypes), dim=-1)

        sols, _, _ = self.proj_ray_pass_to_model(cond, prepped=False)

        return {
            "per_event": {
                "E_dep": sols[:, 1, 1],
                "Density": sols[:, 0, 1],
            },
            "per_particle": {
                "Incoming": sols[:, 2:3],
                "Outgoing": sols[:, 3:],
            },
            "per_voxel": {
                "E_dep": sols.new_empty(sols.shape[0], 0, 4),
            },
        }

    def gen_batch(self, cond: torch.Tensor):
        pdgid_in = cond[:, -1].long()
        pdgid_in_idx = torch.searchsorted(self.pdgid_in, pdgid_in)
        mult = self.gen_mult((cond[:, :-1], None, pdgid_in_idx))
        mult = mult[:, self.valid_ptypes_mask]

        max_particles = self.max_seq_l - 3
        total = mult.sum(-1, keepdim=True)
        scale = (max_particles / total).clamp(max=1.0)
        mult = (mult * scale).long()
        scaled = total > max_particles
        remaining = (scaled * (max_particles - mult.sum(-1, keepdim=True))).clamp(min=0)
        r_max = remaining.max().item()
        if r_max > 0:
            dist = torch.multinomial(mult.float().clamp(min=1), r_max, replacement=True)
            valid = (torch.arange(r_max, device=mult.device) < remaining).long()
            mult.scatter_add_(-1, dist, valid)

        idx = torch.arange(max_particles, device=mult.device)
        attn_mask = idx < mult.sum(-1, keepdim=True)
        pdgid_pad = torch.zeros_like(attn_mask, dtype=torch.long)
        cumsum_idx = mult.cumsum(-1)[..., :-1].clamp(max=max_particles - 1)
        pdgid_pad.scatter_add_(-1, cumsum_idx, torch.ones_like(pdgid_pad)).cumsum_(-1)
        cond[..., -1] = torch.searchsorted(self.pdgids, pdgid_in) + 1
        cond = cond[:, None, :]
        cond_pad_r = cond.repeat(1, self.max_seq_l - 3, 1)
        cond_pad_r[..., -1] = attn_mask * (pdgid_pad + 1)
        cond_fm = torch.cat(
            (torch.zeros_like(cond).expand(-1, 2, -1), cond, cond_pad_r), dim=1
        )

        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :3]), attn_mask), dim=1)
        mask = attn_mask.clone().long().unsqueeze(2)
        mask[:, [0, 2], 0] = 0 #Conditions
        cond_fm[:, 0, 1] = cond[:, 0, 0] #Density
        cond_fm[:, :2, 2:-1] = 1
        return cond_fm, mask, attn_mask
