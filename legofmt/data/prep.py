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

# model_conf.edep_wall: the E_dep token's cap is E_in + a margin [MeV] per incoming species -- what Geant4 can deposit
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
            conf = config["model_conf"]
            self.manifold = build_manifold(conf["manifold"])
            self.proj_ray = conf.get("proj_ray", True)
            cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev")
            max_energy = conf["max_energy"]
        else:
            # manifold is only needed by cc_trafo (the dict path); norm_e-only
            # users (MultLoader) legitimately have no manifold to give.
            conf = config
            self.manifold = build_manifold(conf["manifold"]) if conf.get("manifold") else None
            self.proj_ray = conf.get("proj_ray")
            cutoff_mev = conf.get("cutoff_mev")
            max_energy = conf.get("max_energy")
        self.energy_kin = conf.get("energy_kin", True)
        self.edep_log_min = conf.get("edep_log_min")
        self.edep_wall = conf.get("edep_wall", False)
        self.edep_wall_species = conf.get("edep_wall_species")
        self.edep_excess_mev = conf.get("edep_excess_mev")
        if self.edep_wall and self.edep_excess_mev is not None:
            missing = set(self.edep_wall_species or ()) - set(self.edep_excess_mev)
            if missing:
                raise ValueError(f"edep_wall_species {sorted(missing)} have no entry in model_conf.edep_excess_mev: "
                                 "no such incoming species in the training data, so no cap")
        set_layout(conf.get("cond_scalars", ("Density",)))
        self.pen = EnergyProjections(cutoff_mev=cutoff_mev, max_energy=max_energy)
        self.ppa = CubeTrace(conf.get("cuboid_dim"))

    def __call__(self, batch: tuple) -> Tensor:
        return self.prep(batch)

    @torch.no_grad()
    def prep(self, batch: tuple) -> Tensor:
        """Converts a raw batch ``(cc_ext, mask, attn_mask, data_add)`` to the padded model layout.

        The two max_energy-dependent channels stay in MeV; ``norm_e`` converts them at load time.
        """
        cc_ext, mask, attn_mask, data_add = batch
        e_kin = cc_ext[..., 0:1] if self.energy_kin else None
        model_in = self.cc_trafo(cc_ext[..., 1:7], e_kin=e_kin)
        cc_ext = torch.cat((model_in, cc_ext[..., 7:]), dim=-1)
        return self.format_add((cc_ext, mask, attn_mask, data_add))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        """Converts MeV momenta / positions to (energy scalar, unit momentum direction, unit position direction).

        The result is projected onto the manifold. Column 0 keeps MeV for the incoming token and stores 1 -
        log(e_out/cutoff)/log(e_in/cutoff) for outgoing tokens. The position direction is that of x / cuboid_dim: the
        point on the unit cube the cuboid maps to.
        """
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
        """Converts the two channels that depend on ``max_energy`` from MeV to the model scale.

        A dataset can so be generated without knowing ``max_energy``. The outgoing column is a log ratio: ``cutoff_mev``
        only, already final. Datasets written before the energy split are passed through; see
        ``compat.dataset_already_normalised``.

        Args:
            batch:   ``(f, mask, attn_mask)``; an optional 4th tensor (per-layer E_dep) goes on the E_dep token's scale.
            inplace: write into ``f`` (a freshly loaded file: the copy doubled the load's peak RAM).

        Returns:
            The batch, as new tensors unless ``inplace``.
        """
        f, mask, attn_mask, *edep_layers = batch
        if dataset_already_normalised(f, self.pen.max_energy):
            return batch
        f = f if inplace else f.clone()
        in_cc = _F(f).in_cc
        e_in, pid = in_cc[..., 0, 0].clone(), _F(f).in_p[..., 0, -1]   # MeV, before it becomes the scalar below
        in_cc[..., 0:1] = self.pen.to_scalar_e(in_cc[..., 0:1])
        _F(f).edep.copy_(self.norm_edep_token(_F(f).edep, e_in, pid))
        if edep_layers:   # optional (B, n_layers) tensor; norm_edep is elementwise
            return f, mask, attn_mask, self.norm_edep(edep_layers[0])
        return f, mask, attn_mask

    def norm_edep(self, edep: Tensor) -> Tensor:
        """Converts E_dep from MeV to the model scale.

        E_dep/max_energy, or the edep_log_min log scale clamped to [0, 1].
        """
        if not self.edep_log_min:
            return edep / self.pen.max_energy
        lo = self.edep_log_min
        rng = torch.tensor(self.pen.max_energy / lo).log().item()
        return ((edep / lo).clamp_min(1.0).log() / rng).clamp(0.0, 1.0)

    @staticmethod
    def edep_excess_max(f: Tensor) -> dict[int, float]:
        """Returns, per incoming pdgid, the largest E_dep - E_in [MeV] (>= 0) in a stored file.

        ``f`` holds MeV channels (before ``norm_e``). The result is meta.json's ``edep_excess_max_mev``, which becomes
        ``model_conf.edep_excess_mev``.
        """
        e_in, pid = _F(f).in_cc[..., 0, 0].double(), _F(f).in_p[..., 0, -1]
        excess = _F(f).edep.reshape(e_in.shape).double() - e_in
        return {int(p): max(0.0, excess[pid == p].max().item()) for p in pid.unique().tolist()}

    def _edep_wall_range(self, e_in: Tensor, pid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns E_cap = E_in + margin(species) and the wall's log-odds range.

        The range is [log(eps / E_cap), log(E_cap / eps_missing)].
        """
        margins = self.edep_excess_mev or EDEP_WALL_MARGIN_MEV
        margin = torch.full_like(e_in, max(margins.values()) if self.edep_excess_mev else EDEP_WALL_DEFAULT_MARGIN_MEV)
        for p, m in margins.items():
            margin = torch.where(pid == p, torch.full_like(e_in, m), margin)
        cap = e_in + margin
        return cap, torch.log((self.edep_log_min or 1e-6) / cap), torch.log(cap / EDEP_WALL_EPS_MISSING_MEV)

    def edep_wall_mask(self, pid: Tensor) -> Tensor:
        """Returns where the E_dep token is the wall.

        Nowhere without ``model_conf.edep_wall``; with it for every incoming species, or only for those listed in
        ``model_conf.edep_wall_species`` (raw pdgids; the rest keep the log token).
        """
        if not self.edep_wall:
            return torch.zeros_like(pid, dtype=torch.bool)
        mask = torch.full_like(pid, self.edep_wall_species is None, dtype=torch.bool)
        for p in self.edep_wall_species or ():   # python scalars: no host-to-device copy per call
            mask |= pid == p
        return mask

    def norm_edep_token(self, edep: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep TOKEN from MeV to the model scale.

        ``norm_edep`` unless ``model_conf.edep_wall`` (for the species ``edep_wall_mask`` selects): then the log-odds of
        the deposited fraction of E_cap = E_in + margin(species), z = log E_dep - log(E_cap - E_dep), rescaled from
        [log(eps / E_cap), log(E_cap / eps_missing)] to [0, 1]. Small deposits keep a log scale; near the cap the
        channel resolves the missing energy, so a contained shower's narrow peak below E_in becomes wide, and E_cap is
        the edge u = 1. Zero stays 0 (the overflow sentinel's condition).

        Args:
            edep: the deposit [MeV].
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.
        """
        if not self.edep_wall:
            return self.norm_edep(edep)
        e, (cap, lo, hi) = edep.double(), self._edep_wall_range(e_in.double(), pid)
        z = torch.log(e.clamp_min(1e-300)) - torch.log((cap - e).clamp_min(1e-300))
        u = ((z - lo) / (hi - lo)).clamp(0.0, 1.0)
        u = torch.where(e > 0, u, torch.zeros_like(u)).to(edep.dtype)
        return torch.where(self.edep_wall_mask(pid), u, self.norm_edep(edep))

    def denorm_edep_token(self, u: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Inverts ``norm_edep_token``: E_cap * sigmoid(z); non-positive u (the sentinel) -> 0, u > 1 -> the cap."""
        if not self.edep_wall:
            return self.denorm_edep(u)
        cap, lo, hi = self._edep_wall_range(e_in.double(), pid)
        E = cap * torch.sigmoid(lo + u.double().clamp(0.0, 1.0) * (hi - lo))
        E = torch.where(u > 0, E, torch.zeros_like(E)).to(u.dtype)
        return torch.where(self.edep_wall_mask(pid), E, self.denorm_edep(u))

    def denorm_edep(self, edep: Tensor) -> Tensor:
        """Converts E_dep from the model scale to MeV, the inverse of ``norm_edep`` on ``[0, 1]``.

        Every non-positive input (the ``overflow_delta`` sentinel) decodes to zero.
        """
        if not self.edep_log_min:
            return edep.clamp_min(0.0) * self.pen.max_energy
        lo = self.edep_log_min
        rng = torch.tensor(self.pen.max_energy / lo).log().item()
        return torch.where(edep > 0, lo * (edep.clamp(0.0, 1.0) * rng).exp(),
                           torch.zeros_like(edep))

    @torch.no_grad()
    def format_add(self, batch: tuple) -> Tensor:
        """Prepends the conditioning-scalar tokens and the generated E_dep token.

        mask/attn_mask are extended to match (E_dep is the only prefix token with mask=1). An optional ``E_dep_layers``
        (B, n_layers) entry of data_add is returned as a 4th tensor, in MeV like the E_dep token.
        """
        cc_ext, mask, attn_mask, data_add = batch
        first, *rest = cond_scalars()
        tokens = []
        for name in (first, "E_dep", *rest):
            token = torch.ones_like(cc_ext[:, :1])
            token[..., 0] = data_add[name].view_as(token[..., 0])
            tokens.append(token)
        target = torch.cat((*tokens, cc_ext), dim=1).nan_to_num()
        _F(target).non_p[..., -1] = 0
        z, o = torch.zeros_like(mask[:, :1]), torch.ones_like(mask[:, :1])
        mask = torch.cat((z, o, *([z] * len(rest)), mask), dim=1)
        attn_mask = torch.cat((torch.ones_like(attn_mask[:, :1]).repeat(1, len(tokens)), attn_mask), dim=1)
        if "E_dep_layers" in data_add:
            return target, mask, attn_mask, data_add["E_dep_layers"]
        return target, mask, attn_mask
