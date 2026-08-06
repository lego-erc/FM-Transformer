from dataclasses import replace

import torch
import torch.nn.functional as F

from ..main.modules import LEGOLtng
from ..multiplicity.model import MultModel
from ..geometry.raytracing_proj import CubeTrace
from ..geometry.energy_proj import EnergyProjections
from ..data.struct import _F, DataStruct, set_layout


class GenerateOut(torch.nn.Module):

    flow_cls = LEGOLtng

    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu", couple_in_out_pdgids=False):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = self.flow_cls(flow_conf).to(device)
        self.model.rc = replace(self.model.rc, pdgid_is_idx=True)
        self.cond_names = tuple(self.model.rc.cond_scalars)
        self.n_prefix = self.model.rc.n_prefix
        self.n_cond = len(self.cond_names)  # = n_prefix - 1 (edep excluded)

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        self.pdgid_in = mult_conf["config"]["mm_conf"]["ptypes_in"].to(device)
        self.ptypes = mult_conf["config"]["mm_conf"]["ptypes"].to(device)

        mm_cond = mult_conf["config"]["mm_conf"].get("cond_scalars")
        self.n_mult_cond = len(mm_cond) if mm_cond else self.gen_mult.proj_in_.in_features - 7
        if mm_cond and tuple(mm_cond) != self.cond_names[:self.n_mult_cond]:
            raise ValueError(
                f"mult cond_scalars {tuple(mm_cond)} is not a prefix of the flow's "
                f"{self.cond_names}; cond carries only the flow's scalars."
            )

        set_layout(self.cond_names)

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
        cond_model = cond.clone()
        if not prepped:
            nc = self.n_cond
            mom, pos = cond_model[:, nc:nc + 3], cond_model[:, nc + 3:nc + 6]
            pos = F.normalize(
                self.proj_ray(torch.cat((mom, pos), dim=-1))[..., 3:], dim=-1
            )
            dir_, e = self.pen.to_scalar(mom)
            cond_model = torch.cat(
                (cond_model[:, :nc], e, dir_, pos, cond_model[:, nc + 6:]), dim=-1
            )
        batch = self.gen_batch(cond_model)
        sols, mask, attn_mask = self.model(batch)
        sols[..., -1] = torch.cat([sols.new_zeros(1), self.pdgids.to(sols.dtype)])[sols[..., -1].long()]
        return sols, mask, attn_mask

    def gen_model_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None):
        device = next(self.model.parameters()).device
        pos, mom, energy, density, size, pdgids = (
            t.to(device) for t in (pos, mom, energy, density, size, pdgids)
        )
        Z = Z.to(device) if Z is not None else None
        A = A.to(device) if A is not None else None

        scalar_src = {"Density": density, "Z": Z, "A": A, "Size": size}
        missing = [k for k in self.cond_names if scalar_src.get(k) is None]
        if missing:
            raise ValueError(f"gen_model_w_g4_args missing conditioning arrays for {missing}")
        if "Size" not in self.cond_names and size.shape[0] != 1:
            raise ValueError("Multiple sizes not yet supported (size is not a conditioning variable).")

        shapes = {
            "pos": pos.view(-1, 3).shape[0],
            "mom": mom.view(-1, 3).shape[0],
            "energy": energy.shape[0],
            "pdgids": pdgids.shape[0],
            **{k: scalar_src[k].reshape(-1).shape[0] for k in self.cond_names},
        }
        B = max(shapes.values())
        err_size = {k: v for k, v in shapes.items() if v not in (1, B)}
        if err_size:
            raise ValueError(
                f"Each argument must have either size 1 or batch size {B}; got {err_size}"
            )

        mom = F.normalize(mom, dim=-1)

        def _scalar_col(x):
            return x.reshape(-1, 1).expand(B, 1).repeat_interleave(n, 0)

        e = energy.view(-1, 1).expand(B, 1)
        mom_b = mom.view(-1, 3).expand(B, 3)
        pos_b = pos.view(-1, 3).expand(B, 3)
        cc = torch.cat((mom_b * e, pos_b), dim=-1).repeat_interleave(n, dim=0)

        conds_b = torch.cat([_scalar_col(scalar_src[k]) for k in self.cond_names], dim=-1)
        pdgids_b = _scalar_col(pdgids).to(cc.dtype)
        cond = torch.cat((conds_b, cc, pdgids_b), dim=-1)

        sols, _, _ = self(cond)
        s = _F(sols)
        per_event = {"E_dep": s.edep}
        per_event.update({k: s.cond(k) for k in self.cond_names})
        return {
            "per_event": per_event,
            "per_particle": {"Incoming": s.in_p, "Outgoing": s.out_p},
            "per_voxel": {"E_dep": sols.new_empty(sols.shape[0], 0, 4)},
        }

    def gen_batch(self, cond: torch.Tensor):
        pdgid_in = cond[:, -1].long()
        pdgid_in_idx = torch.searchsorted(self.pdgid_in, pdgid_in)
        mult_in = torch.cat(
            (cond[:, :self.n_mult_cond], cond[:, self.n_cond:self.n_cond + 7]), dim=-1
        )
        mult = self.gen_mult((mult_in, None, pdgid_in_idx))
        mult = mult[:, self.ptype_idx] * self.ptype_in_mask

        max_particles = self.max_seq_l - (self.n_prefix + 1)
        total = mult.sum(-1, keepdim=True)
        scale = (max_particles / total).clamp(max=1.0)
        mult = (mult * scale).long()
        scaled = total > max_particles
        remaining = (scaled * (max_particles - mult.sum(-1, keepdim=True))).clamp(min=0)
        dist = torch.multinomial(mult.float().clamp(min=1), max_particles, replacement=True)
        valid = (torch.arange(max_particles, device=mult.device) < remaining).long()
        mult.scatter_add_(-1, dist, valid)

        idx = torch.arange(max_particles, device=mult.device)
        occupied = idx < mult.sum(-1, keepdim=True)
        pdgid_pad = torch.zeros_like(occupied, dtype=torch.long)
        cumsum_idx = mult.cumsum(-1)[..., :-1].clamp(max=max_particles - 1)
        pdgid_pad.scatter_add_(-1, cumsum_idx, torch.ones_like(pdgid_pad)).cumsum_(-1)

        conds = cond[:, :self.n_cond]          # per-event conditioning scalars
        in_tok = cond[:, self.n_cond:]         # [e, dir(3), pos(3), pdgid] (8 cols)
        in_tok[..., -1] = torch.searchsorted(self.pdgids, pdgid_in) + 1
        in_tok = in_tok[:, None, :]
        out_tok = in_tok.repeat(1, max_particles, 1)
        out_tok[..., -1] = occupied * (pdgid_pad + 1)
        cond_fm = torch.cat(
            (torch.zeros_like(in_tok).expand(-1, self.n_prefix, -1), in_tok, out_tok), dim=1
        )

        attn_mask = torch.cat(
            (torch.ones_like(occupied[:, :1]).repeat(1, self.n_prefix + 1), occupied), dim=1
        )
        mask = attn_mask.clone().long()
        cond_idx = [0, *range(2, self.n_prefix), self.n_prefix]  # cond scalars + incoming; edep stays 1
        mask[:, cond_idx] = 0
        cond_fm[:, 0, 0] = conds[:, 0]
        for j in range(1, self.n_cond):
            cond_fm[:, j + 1, 0] = conds[:, j]
        _F(cond_fm).non_p[..., 1:-1] = 1
        return cond_fm, mask, attn_mask


class GenerateIn(GenerateOut):

    @torch.no_grad()
    def __call__(self, batch):
        ds = batch if isinstance(batch, DataStruct) else DataStruct(*batch)
        f = ds.f.full.clone()
        m = torch.zeros_like(ds.m.full)
        m[:, self.n_prefix] = 1  # only the incoming slot is inferred
        ds = DataStruct(f, m, ds.am.full)
        conds = torch.cat(
            [ds.f.cond(n).view(-1, 1, 1) for n in self.cond_names], dim=-1
        ).expand(-1, ds.f.out_cc.shape[1], -1)
        out_tok = torch.cat((conds, ds.f.out_cc), dim=-1)
        out_pid = torch.searchsorted(
            self.ptypes, ds.f.out_p[..., -1].long()
        ).clamp(max=len(self.ptypes) - 1)
        pid_in_idx = self.gen_mult((out_tok, out_pid, ds.am.out_p.bool(), ds.f.edep))
        # The flow runs with pdgid_is_idx=True: swap the raw ids the mult
        # model consumed above for flow-vocabulary indices.
        f[..., -1] = self.model.convert_pdgids(f[..., -1]).to(f.dtype)
        f[:, self.n_prefix, -1] = torch.searchsorted(
            self.pdgids, self.pdgid_in[pid_in_idx]
        ).clamp(max=len(self.pdgids) - 1) + 1
        sols, _, _ = self.model(ds)
        sols[..., -1] = torch.cat(
            [sols.new_zeros(1), self.pdgids.to(sols.dtype)]
        )[sols[..., -1].long()]
        return _F(sols).in_p

