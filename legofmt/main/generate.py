import torch
import torch.nn.functional as F

from ..main.modules import LEGOLtng, LEGOLtngDirect
from ..multiplicity.model import MultModel
from ..geometry.raytracing_proj import CubeTrace
from ..geometry.energy_proj import EnergyProjections
from ..data.struct import _F


class GenerateOut(torch.nn.Module):
    flow_cls = LEGOLtng

    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu", couple_in_out_pdgids=False):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = self.flow_cls(flow_conf).to(device)
        object.__setattr__(self.model.rc, "pdgid_is_idx", True)

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        self.pdgid_in = mult_conf["config"]["mm_conf"]["ptypes_in"].to(device)
        self.ptypes = mult_conf["config"]["mm_conf"]["ptypes"].to(device)

        if couple_in_out_pdgids:
            self.model.rc.odeint_conf["filter_pdgid"] = self.pdgid_in

        self.max_seq_l = flow_conf["config"]["model_conf"]["model_args"]["max_seq_l"]
        self.pdgids = flow_conf["config"]["model_conf"]["pdgids"].to(device)
        self.ptype_idx = torch.searchsorted(self.ptypes, self.pdgids).clamp(max=len(self.ptypes) - 1)
        self.ptype_in_mask = self.ptypes[self.ptype_idx] == self.pdgids
        self.proj_ray = CubeTrace()
        self.pen = EnergyProjections(
            cutoff_mev=self.model.rc.cutoff_mev, max_energy=self.model.rc.max_energy,
        )

    def __call__(self, cond: torch.Tensor, prepped: bool = False):
        model_out = self.proj_ray_pass_to_model(cond, prepped=prepped)
        return model_out

    def proj_ray_pass_to_model(self, cond: torch.Tensor, prepped: bool = False):
        cond_model = cond.clone()
        if not prepped:
            mom, pos = cond_model[:, 1:4], cond_model[:, 4:7]
            pos = self.proj_ray(torch.cat((mom, pos), dim=-1))[..., 3:]
            dir_, e = self.pen.to_scalar(mom)
            cond_model = torch.cat(
                (cond_model[:, :1], e, dir_, pos, cond_model[:, 7:]), dim=-1
            )
        batch = self.gen_batch(cond_model)
        sols, mask, attn_mask = self.model(batch)
        sols[..., -1] = torch.cat([sols.new_zeros(1), self.pdgids.to(sols.dtype)])[sols[..., -1].long()]
        return sols, mask, attn_mask
    
    def gen_model_w_g4_args(self, n, pos, mom, energy, density, size, pdgids):
        device = next(self.model.parameters()).device
        pos, mom, energy, density, size, pdgids = (
            t.to(device) for t in (pos, mom, energy, density, size, pdgids)
        )

        if size.shape[0] != 1:
            raise ValueError("Multiple sizes not yet supported.")
    
        shapes = {
            "pos": pos.view(-1, 3).shape[0],                                                                                                       
            "mom": mom.view(-1, 3).shape[0],                   
            "energy": energy.shape[0],                                                                                                             
            "density": density.shape[0],
            "pdgids": pdgids.shape[0],                                                                                                             
        }

        B = max(shapes.values())
        err_size = {k: v for k, v in shapes.items() if v not in (1, B)}                                                                                 
        if err_size:
            raise ValueError(                                                                                                                      
                f"Each argument must have either size 1 or batch size {B}; got {err_size}"
            )
        
        mom = F.normalize(mom, dim=-1)

        if all([shape == B for shape in shapes.values()]):
            cc = torch.cat((mom.view(-1, 3) * energy.view(-1, 1), pos.view(-1, 3)), dim=-1)
            cc = cc.repeat_interleave(n, dim=0)
            d = density.view(-1, 1).repeat_interleave(n, dim=0)
            pdgids_b = pdgids.view(-1, 1).repeat_interleave(n, dim=0).to(cc.dtype)
        else:
            e = energy.view(-1, 1).expand(B, 1)
            mom_b = mom.view(-1, 3).expand(B, 3)
            pos_b = pos.view(-1, 3).expand(B, 3)
            d = density.view(-1, 1).expand(B, 1).repeat_interleave(n, dim=0)
            pdgids_b = pdgids.view(-1, 1).expand(B, 1).repeat_interleave(n, dim=0)
            cc = torch.cat((mom_b * e, pos_b), dim=-1).repeat_interleave(n, dim=0)

        cond = torch.cat((d, cc, pdgids_b), dim=-1)

        sols, _, _ = self.proj_ray_pass_to_model(cond, prepped=False)
        s = _F(sols)
        return {
            "per_event": {"E_dep": s.edep, "Density": s.d},
            "per_particle": {"Incoming": s.in_p, "Outgoing": s.out_p},
            "per_voxel": {"E_dep": sols.new_empty(sols.shape[0], 0, 4)},
        }

    def gen_batch(self, cond: torch.Tensor):
        pdgid_in = cond[:, -1].long()
        pdgid_in_idx = torch.searchsorted(self.pdgid_in, pdgid_in)
        mult = self.gen_mult((cond[:, :-1], None, pdgid_in_idx))
        mult = mult[:, self.ptype_idx] * self.ptype_in_mask

        max_particles = self.max_seq_l - 3
        total = mult.sum(-1, keepdim=True)
        scale = (max_particles / total).clamp(max=1.0)
        mult = (mult * scale).long()
        scaled = total > max_particles
        remaining = (scaled * (max_particles - mult.sum(-1, keepdim=True))).clamp(min=0)
        dist = torch.multinomial(mult.float().clamp(min=1), max_particles, replacement=True)
        valid = (torch.arange(max_particles, device=mult.device) < remaining).long()
        mult.scatter_add_(-1, dist, valid)

        idx = torch.arange(max_particles, device=mult.device)
        attn_mask = idx < mult.sum(-1, keepdim=True)
        pdgid_pad = torch.zeros_like(attn_mask, dtype=torch.long)
        cumsum_idx = mult.cumsum(-1)[..., :-1].clamp(max=max_particles - 1)
        pdgid_pad.scatter_add_(-1, cumsum_idx, torch.ones_like(pdgid_pad)).cumsum_(-1)
        density = cond[:, 0]
        token = cond[:, 1:]
        token[..., -1] = torch.searchsorted(self.pdgids, pdgid_in) + 1
        token = token[:, None, :]
        cond_pad_r = token.repeat(1, self.max_seq_l - 3, 1)
        cond_pad_r[..., -1] = attn_mask * (pdgid_pad + 1)
        cond_fm = torch.cat(
            (torch.zeros_like(token).expand(-1, 2, -1), token, cond_pad_r), dim=1
        )

        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :3]), attn_mask), dim=1)
        mask = attn_mask.clone().long()
        mask[:, [0, 2]] = 0
        cond_fm[:, 0, 0] = density
        _F(cond_fm).non_p[..., 1:-1] = 1
        return cond_fm, mask, attn_mask


class GenerateOutDirect(GenerateOut):
    """Direct variant — uses :class:`legofmt.main.modules.LEGOLtngDirect`
    as the flow component."""
    flow_cls = LEGOLtngDirect
