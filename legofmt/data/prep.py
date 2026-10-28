"""``DataPrep``: raw MeV particles -> the model's normalised column layout.

Applies the energy normalisation, the cube ray-trace, and the outgoing-relative
energy transform (``cc_trafo``). Accepts either a full config or the flat dict
``MultLoader`` passes, which carries no manifold.
"""


import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.compat import dataset_already_normalised
from legofmt.data.struct import _F, cond_scalars, set_layout
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.mod_comps.config import build_manifold

# model_conf.edep_wall: the E_dep row's cap is E_in + a margin [MeV] per incoming species -- what Geant4 can deposit
# above E_in. Normally the training data's own largest excess (model_conf.edep_excess_mev, DataPrep.edep_excess_max);
# these hand-set values only serve checkpoints from before it (e+ annihilation; pions: absorption releases the rest
# mass, up to +178 MeV in the stack data; neutrons: binding, 3e-5 of events by up to 4.6 MeV). Species not listed get
# the widest margin.
EDEP_WALL_MARGIN_MEV = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.0, 2112: 10.0, 211: 200.0, -211: 200.0}
EDEP_WALL_DEFAULT_MARGIN_MEV = 200.0
EDEP_WALL_EPS_MISSING_MEV = 1e-3   # the smallest missing energy E_cap - E_dep the channel resolves


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
            self.edep_log_min = model_conf.get("edep_log_min")
            self.edep_wall = model_conf.get("edep_wall", False)
            self.edep_wall_species = model_conf.get("edep_wall_species")
            self.edep_excess_mev = model_conf.get("edep_excess_mev")
            cuboid_dim = model_conf.get("cuboid_dim")
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
            self.edep_log_min = config.get("edep_log_min")
            self.edep_wall = config.get("edep_wall", False)
            self.edep_wall_species = config.get("edep_wall_species")
            self.edep_excess_mev = config.get("edep_excess_mev")
            cuboid_dim = config.get("cuboid_dim")
        if self.edep_wall and self.edep_excess_mev is not None:
            missing = set(self.edep_wall_species or ()) - set(self.edep_excess_mev)
            if missing:
                raise ValueError(f"edep_wall_species {sorted(missing)} have no entry in model_conf.edep_excess_mev: "
                                 "no such incoming species in the training data, so no cap")
        set_layout(cond)
        self.pen = EnergyProjections(cutoff_mev=cutoff_mev, max_energy=max_energy)
        self.ppa = CubeTrace(cuboid_dim)

    def __call__(self, batch: tuple) -> Tensor:
        return self.prep(batch)

    @torch.no_grad()
    def prep(self, batch: tuple) -> Tensor:
        """Raw (cc_ext, mask, attn_mask, data_add) -> the padded model layout.
        The two max_energy-dependent channels stay in MeV; norm_e converts them
        at load time."""
        cc_ext, mask, attn_mask, data_add = batch
        e_kin = cc_ext[..., 0:1] if self.energy_kin else None
        model_in = self.cc_trafo(cc_ext[..., 1:7], e_kin=e_kin)
        cc_ext = torch.cat((model_in, cc_ext[..., 7:]), dim=-1)
        return self.format_add((cc_ext, mask, attn_mask, data_add))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        """MeV momenta/positions -> (energy scalar, unit mom dir, unit pos dir)
        projected onto the manifold; col 0 keeps MeV for the incoming row and
        stores 1 - log(e_out/cutoff)/log(e_in/cutoff) for outgoing rows. The pos
        dir is that of x / cuboid_dim: the point on the unit cube the cuboid maps to."""
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
            torch.cat((e, dir_, F.normalize(pos / self.ppa.cuboid_dim.to(pos), dim=-1)), dim=-1)
        )

    @torch.no_grad()
    def norm_e(self, batch: tuple, inplace: bool = False) -> tuple:
        """MeV -> model scale for the two channels that depend on ``max_energy``,
        so a dataset can be generated without knowing it. The outgoing column is
        a log ratio: ``cutoff_mev`` only, already final. An optional 4th tensor
        (per-layer E_dep) goes on the E_dep row's scale. Returns new tensors unless
        ``inplace`` (a freshly loaded file: the copy doubled the load's peak RAM).

        Datasets written before the energy split are passed through; see
        ``compat.dataset_already_normalised``.
        """
        f, mask, attn_mask, *edep_layers = batch
        if dataset_already_normalised(f, self.pen.max_energy):
            return batch
        f = f if inplace else f.clone()
        in_cc = _F(f).in_cc
        e_in, pid = in_cc[..., 0, 0].clone(), _F(f).in_p[..., 0, -1]   # MeV, before it becomes the scalar below
        in_cc[..., 0:1] = self.pen.to_scalar_e(in_cc[..., 0:1])
        _F(f).edep.copy_(self.norm_edep_row(_F(f).edep, e_in, pid))
        if edep_layers:   # optional (B, n_layers) tensor; norm_edep is elementwise
            return f, mask, attn_mask, self.norm_edep(edep_layers[0])
        return f, mask, attn_mask

    def norm_edep(self, edep: Tensor) -> Tensor:
        """MeV -> model scale for E_dep: E_dep/max_energy, or the edep_log_min
        log scale clamped to [0, 1]."""
        if not self.edep_log_min:
            return edep / self.pen.max_energy
        lo = self.edep_log_min
        rng = torch.tensor(self.pen.max_energy / lo).log().item()
        return ((edep / lo).clamp_min(1.0).log() / rng).clamp(0.0, 1.0)

    @staticmethod
    def edep_excess_max(f: Tensor) -> dict[int, float]:
        """Per incoming pdgid, the largest E_dep - E_in [MeV] (>= 0) in a stored file (MeV channels, before
        ``norm_e``): meta.json's ``edep_excess_max_mev``, which becomes ``model_conf.edep_excess_mev``."""
        e_in, pid = _F(f).in_cc[..., 0, 0].double(), _F(f).in_p[..., 0, -1]
        excess = _F(f).edep.reshape(e_in.shape).double() - e_in
        return {int(p): max(0.0, excess[pid == p].max().item()) for p in pid.unique().tolist()}

    def _edep_cap(self, e_in: Tensor, pid: Tensor) -> Tensor:
        margins = self.edep_excess_mev or EDEP_WALL_MARGIN_MEV
        margin = torch.full_like(e_in, max(margins.values()) if self.edep_excess_mev else EDEP_WALL_DEFAULT_MARGIN_MEV)
        for p, m in margins.items():
            margin = torch.where(pid == p, torch.full_like(e_in, m), margin)
        return e_in + margin

    def _edep_wall_range(self, cap: Tensor) -> tuple[Tensor, Tensor]:
        return torch.log((self.edep_log_min or 1e-6) / cap), torch.log(cap / EDEP_WALL_EPS_MISSING_MEV)

    def edep_wall_mask(self, pid: Tensor) -> Tensor:
        """Where the E_dep row is the wall: nowhere without ``model_conf.edep_wall``, for every incoming species with
        it, or only for those listed in ``model_conf.edep_wall_species`` (raw pdgids; the rest keep the log row)."""
        if not self.edep_wall:
            return torch.zeros_like(pid, dtype=torch.bool)
        mask = torch.full_like(pid, self.edep_wall_species is None, dtype=torch.bool)
        for p in self.edep_wall_species or ():   # python scalars: no host-to-device copy per call
            mask |= pid == p
        return mask

    def norm_edep_row(self, edep: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """MeV -> model scale for the E_dep ROW, given the incoming kinetic energy [MeV] and raw pdgid (same shape).
        ``norm_edep`` unless ``model_conf.edep_wall`` (for the species ``edep_wall_mask`` selects): then the
        log-odds of the deposited fraction of E_cap = E_in + margin(species), z = log E_dep - log(E_cap - E_dep),
        rescaled from [log(eps / E_cap), log(E_cap / eps_missing)] to [0, 1]. Small deposits keep a log scale; near
        the cap the channel resolves the missing energy, so a contained shower's narrow peak below E_in becomes
        wide, and E_cap is the edge u = 1. Zero stays 0 (the overflow sentinel's condition)."""
        if not self.edep_wall:
            return self.norm_edep(edep)
        e, cap = edep.double(), self._edep_cap(e_in.double(), pid)
        lo, hi = self._edep_wall_range(cap)
        z = torch.log(e.clamp_min(1e-300)) - torch.log((cap - e).clamp_min(1e-300))
        u = ((z - lo) / (hi - lo)).clamp(0.0, 1.0)
        u = torch.where(e > 0, u, torch.zeros_like(u)).to(edep.dtype)
        return torch.where(self.edep_wall_mask(pid), u, self.norm_edep(edep))

    def denorm_edep_row(self, u: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Inverse of ``norm_edep_row``: E_cap * sigmoid(z); non-positive u (the sentinel) -> 0, u > 1 -> the cap."""
        if not self.edep_wall:
            return self.denorm_edep(u)
        cap = self._edep_cap(e_in.double(), pid)
        lo, hi = self._edep_wall_range(cap)
        E = cap * torch.sigmoid(lo + u.double().clamp(0.0, 1.0) * (hi - lo))
        E = torch.where(u > 0, E, torch.zeros_like(E)).to(u.dtype)
        return torch.where(self.edep_wall_mask(pid), E, self.denorm_edep(u))

    def denorm_edep(self, edep: Tensor) -> Tensor:
        """Model scale -> MeV, the inverse of ``norm_edep`` on ``[0, 1]``. Every
        non-positive input (the ``overflow_delta`` sentinel) decodes to zero."""
        if not self.edep_log_min:
            return edep.clamp_min(0.0) * self.pen.max_energy
        lo = self.edep_log_min
        rng = torch.tensor(self.pen.max_energy / lo).log().item()
        return torch.where(edep > 0, lo * (edep.clamp(0.0, 1.0) * rng).exp(),
                           torch.zeros_like(edep))

    @torch.no_grad()
    def format_add(self, batch: tuple) -> Tensor:
        """Prepend the conditioning-scalar rows and the generated E_dep row,
        extending mask/attn_mask to match (E_dep is the only prefix row with
        mask=1). An optional ``E_dep_layers`` (B, n_layers) entry of data_add is
        returned as a 4th tensor, in MeV like the E_dep row."""
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
        if "E_dep_layers" in data_add:
            return target, mask, attn_mask, data_add["E_dep_layers"]
        return target, mask, attn_mask
