
import torch
import torch.nn.functional as F
from torch import Tensor

from ..compat import dataset_already_normalised
from ..geometry.energy_proj import EnergyProjections
from ..geometry.raytracing_proj import CubeTrace
from ..mod_comps.config import build_manifold
from .struct import _F, cond_scalars, set_layout

class DataPrep:
    def __init__(self, config):
        config = config.get("config", config)
        if "model_conf" in config:
            model_conf = config["model_conf"]
            self.manifold = build_manifold(model_conf["manifold"])
            self.proj_ray = model_conf.get("proj_ray", True)
            cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev")
            max_energy = model_conf["max_energy"]
            cond = model_conf.get("cond_scalars", ("Density",))
            self.energy_kin = model_conf.get("energy_kin", True)
        else:
            # manifold is only needed by cc_trafo (the dict path); norm_e-only
            # users (MultLoader) legitimately have no manifold to give.
            m = config.get("manifold")
            self.manifold = build_manifold(m) if m else None
            self.proj_ray = config.get("proj_ray")
            cutoff_mev = config.get("cutoff_mev")
            max_energy = config.get("max_energy")
            cond = config.get("cond_scalars", ("Density",))
            self.energy_kin = config.get("energy_kin", True)
        set_layout(cond)
        self.pen = EnergyProjections(cutoff_mev=cutoff_mev, max_energy=max_energy)
        self.ppa = CubeTrace()

    def __call__(self, batch: tuple) -> Tensor:
        return self.prep(batch)

    @torch.no_grad()
    def prep(self, batch: tuple) -> Tensor:
        cc_ext, mask, attn_mask, data_add = batch
        e_kin = cc_ext[..., 0:1] if self.energy_kin else None
        model_in = self.cc_trafo(cc_ext[..., 1:7], e_kin=e_kin)
        cc_ext = torch.cat((model_in, cc_ext[..., 7:]), dim=-1)
        return self.format_add((cc_ext, mask, attn_mask, data_add))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        cc = cc.nan_to_num(1)
        mom, pos = cc.split(3, -1)
        dir_ = F.normalize(mom, dim=-1)
        # Energy in MeV: Geant4's recorded kinetic energy, else |p| (which
        # saturates for hadrons above T ~ 433 MeV).
        e_mev = e_kin.nan_to_num(1) if e_kin is not None \
            else mom.norm(dim=-1, keepdim=True)
        lg = (e_mev.clamp_min(1e-8) / self.pen.cutoff).log()
        e = torch.cat(
            (e_mev[:, :1], 1 - (lg[:, 1:] / lg[:, :1].clamp_min(1e-6)).clamp(0, 1)), dim=1,
        )
        if self.proj_ray:
            ray = torch.cat((dir_[:, 0], pos[:, 0]), dim=-1)
            pos = pos.clone()
            pos[:, 0] = self.ppa(ray)[..., 3:]
        return self.manifold.projx(
            torch.cat((e, dir_, F.normalize(pos, dim=-1)), dim=-1)
        )

    @torch.no_grad()
    def norm_e(self, batch: tuple) -> tuple:
        """MeV -> model scale for the two channels that depend on ``max_energy``,
        so a dataset can be generated without knowing it. The outgoing column is
        a log ratio: ``cutoff_mev`` only, already final. Returns a new tensor.

        Datasets written before the energy split are passed through; see
        ``compat.dataset_already_normalised``.
        """
        f, mask, attn_mask = batch
        if dataset_already_normalised(f, self.pen.max_energy):
            return batch
        f = f.clone()
        in_cc = _F(f).in_cc
        in_cc[..., 0:1] = self.pen.to_scalar_e(in_cc[..., 0:1])
        _F(f).edep.div_(self.pen.max_energy)
        return f, mask, attn_mask

    @torch.no_grad()
    def format_add(self, batch: tuple) -> Tensor:
        cc_ext, mask, attn_mask, data_add = batch
        e_dep = torch.ones_like(cc_ext[:, :1])
        e_dep[..., 0] = data_add["E_dep"].view_as(e_dep[..., 0])
        cond_rows = []
        for name in cond_scalars():
            row = torch.ones_like(cc_ext[:, :1])
            row[..., 0] = data_add[name].view_as(row[..., 0])
            cond_rows.append(row)
        target = torch.cat((cond_rows[0], e_dep, *cond_rows[1:], cc_ext), dim=1).nan_to_num()
        _F(target).non_p[..., -1] = 0
        z, o = torch.zeros_like(mask[:, :1]), torch.ones_like(mask[:, :1])
        mask = torch.cat((z, o, *([z] * (len(cond_rows) - 1)), mask), dim=1)
        attn_mask = torch.cat(
            (torch.ones_like(attn_mask[:, :1]).repeat(1, len(cond_rows) + 1), attn_mask), dim=1,
        )
        return target, mask, attn_mask
