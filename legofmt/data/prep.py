"""``DataPrep``: raw MeV particles -> the model's normalised tokens.

A raw particle is ``[E_kin, p (3), x (3), pdgid]`` (MeV, mm). ``prep`` writes what a dataset file stores; ``norm_e``
puts the two channels that depend on ``max_energy`` on the model scale when the file is loaded, so a dataset is
generated without knowing ``max_energy``.

```
energy     incoming   e_in [MeV]  --norm_e-->  log(e_in / cutoff) / log(max_energy / cutoff)
           outgoing   1 - log(e_out / cutoff) / log(e_in / cutoff)
momentum              p / |p|
position              direction of x / cuboid_dim             incoming: x ray-traced onto the cube first

E_dep      log scale      u = log(edep / e_min) / log(max_energy / e_min)        e_min = edep_log_min
           margin token   z = log(edep) - log(cap - edep)                        cap  = e_in + margin(species)
                          u = (z - z_lo) / (z_hi - z_lo)                         z_lo = log(e_min / cap)
                                                                                 z_hi = log(cap / 1e-3 MeV)
```

Every channel lands in [0, 1]. The margin token (``model_conf.edep_margin``) is the log-odds of the deposited
fraction of the cap: a small deposit keeps a log scale, and near the cap the channel resolves the missing energy
``cap - edep``, so a contained shower's narrow peak below ``e_in`` becomes wide. ``margin(species)`` is the training
data's largest ``edep - e_in`` (``model_conf.edep_excess_mev``). An incoming species not listed in
``model_conf.edep_margin_species`` keeps the log scale, and so do the per-layer deposits.

``DataPrep`` takes a full config, or the flat dict of ``scripts/gen_data.py`` and ``MultLoader`` (without a manifold:
it only calls ``norm_e``). Superseded: the linear E_dep scale and hand-set margins (:mod:`legofmt.legacy.prep`),
``energy_kin: false`` and ``proj_ray: false`` (one line each in ``cc_trafo``).
"""


import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.compat import dataset_already_normalised
from legofmt.data.struct import _F, cond_scalars, set_layout
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.legacy.prep import HAND_SET_MARGIN_MEV, denorm_edep_linear, norm_edep_linear
from legofmt.mod_comps.config import build_manifold

_MISSING_MEV = 1e-3     # the smallest missing energy cap - edep the margin token resolves
_TINY        = 1e-300   # holds the margin token's two logs finite at edep = 0 and edep = cap


class DataPrep:
    """Converts raw MeV particles to the model's tokens (``prep``, ``norm_e``), and E_dep to and from its scales."""

    def __init__(self, config):
        config = config.get("config", config)
        if "model_conf" in config:   # a full config
            conf = config["model_conf"]
            self.manifold = build_manifold(conf["manifold"])
            self.proj_ray = conf.get("proj_ray", True)
            cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev")
            max_energy = conf["max_energy"]
        else:                        # the flat dict: only cc_trafo needs the manifold, MultLoader has none to give
            conf = config
            self.manifold = build_manifold(conf["manifold"]) if conf.get("manifold") else None
            self.proj_ray = conf.get("proj_ray")
            cutoff_mev = conf.get("cutoff_mev")
            max_energy = conf.get("max_energy")
        self.energy_kin = conf.get("energy_kin", True)
        self.edep_log_min = conf.get("edep_log_min")
        self.edep_margin = conf.get("edep_margin", conf.get("edep_wall", False))                   # edep_wall*: the
        self.edep_margin_species = conf.get("edep_margin_species", conf.get("edep_wall_species"))  # keys until 11.2026
        self.edep_excess_mev = conf.get("edep_excess_mev")
        if self.edep_margin and self.edep_excess_mev is not None:
            missing = set(self.edep_margin_species or ()) - set(self.edep_excess_mev)
            if missing:
                raise ValueError(f"edep_margin_species {sorted(missing)} have no entry in model_conf.edep_excess_mev: "
                                 "no such incoming species in the training data, so no cap")
        set_layout(conf.get("cond_scalars", ("Density",)))
        self.pen = EnergyProjections(cutoff_mev=cutoff_mev, max_energy=max_energy)
        self.ppa = CubeTrace(conf.get("cuboid_dim"))

    def __call__(self, batch: tuple) -> tuple:
        return self.prep(batch)

    # ---- raw particles -> stored tokens -------------------------------------------------------------------------

    @torch.no_grad()
    def prep(self, batch: tuple) -> tuple:
        """Converts a raw batch to the padded model layout, as a dataset file stores it.

        The two ``max_energy``-dependent channels (incoming energy, E_dep) stay in MeV; ``norm_e`` converts them at
        load time.

        Args:
            batch: ``(cc_ext, mask, attn_mask, data_add)`` as ``GetLEGOData`` returns it: (B, 1 + n_out, 8) raw
                particles, the incoming one first, and the per-event scalars.

        Returns:
            ``(f, mask, attn_mask)``, with the per-layer E_dep [MeV] as a 4th tensor when ``data_add`` carries it.
        """
        cc_ext, mask, attn_mask, data_add = batch
        e_kin = cc_ext[..., 0:1] if self.energy_kin else None
        model_in = self.cc_trafo(cc_ext[..., 1:7], e_kin=e_kin)
        cc_ext = torch.cat((model_in, cc_ext[..., 7:]), dim=-1)
        return self.format_add((cc_ext, mask, attn_mask, data_add))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        """Converts MeV momenta / positions to (energy, unit momentum direction, unit position direction).

        ```
        energy     incoming   e_in [MeV]
                   outgoing   1 - log(e_out / cutoff) / log(e_in / cutoff)
        momentum              p / |p|
        position              direction of x / cuboid_dim: the point on the unit cube the cuboid maps to
        ```

        Args:
            cc:    (B, 1 + n_out, 6), momentum and position per particle, the incoming one first; NaN is padding.
            e_kin: (B, 1 + n_out, 1), Geant4's recorded kinetic energy [MeV].

        Returns:
            (B, 1 + n_out, 7), on the manifold.
        """
        mom, pos = cc.nan_to_num(1).split(3, -1)
        e_mev = (e_kin.nan_to_num(1) if e_kin is not None
                 else mom.norm(dim=-1, keepdim=True))   # energy_kin false (legacy): |p|, saturates for hadrons

        log_e   = torch.log(e_mev.clamp_min(1e-8) / self.pen.cutoff)
        e_out   = 1 - (log_e[:, 1:] / log_e[:, :1].clamp_min(1e-6)).clamp(0, 1)
        energy  = torch.cat((e_mev[:, :1], e_out), dim=1)
        mom_dir = F.normalize(mom, dim=-1)
        if self.proj_ray:   # the incoming position, traced along its momentum onto the cube; false: legacy
            pos = pos.clone()
            pos[:, 0] = self.ppa(torch.cat((mom_dir[:, 0], pos[:, 0]), dim=-1))[..., 3:]
        pos_dir = F.normalize(pos / self.ppa.cuboid_dim.to(pos), dim=-1)
        return self.manifold.projx(torch.cat((energy, mom_dir, pos_dir), dim=-1))

    @torch.no_grad()
    def format_add(self, batch: tuple) -> tuple:
        """Prepends the conditioning-scalar tokens and the generated E_dep token (the layout of ``struct.py``).

        Args:
            batch: ``(cc_ext, mask, attn_mask, data_add)``: (B, 1 + n_out, 8) particle tokens, their masks, and the
                per-event scalars by name (``E_dep``, the ``cond_scalars``, optionally ``E_dep_layers``).

        Returns:
            ``(f, mask, attn_mask)`` with the prefix tokens in front (E_dep is the only one with ``mask = 1``); the
            (B, n_layers) ``E_dep_layers`` as a 4th tensor, in MeV like the E_dep token.
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

    # ---- stored tokens -> the model scale -----------------------------------------------------------------------

    @torch.no_grad()
    def norm_e(self, batch: tuple, inplace: bool = False) -> tuple:
        """Converts the two channels that depend on ``max_energy`` from MeV to the model scale.

        The incoming energy becomes ``log(e_in / cutoff) / log(max_energy / cutoff)``, the E_dep token its scale
        (``norm_edep_token``). The outgoing energy is a log ratio: ``cutoff_mev`` only, already final. A dataset
        written before the energy split is passed through; see ``compat.dataset_already_normalised``.

        Args:
            batch:   ``(f, mask, attn_mask)``; an optional 4th tensor (per-layer E_dep) goes on the log scale.
            inplace: write into ``f`` (a freshly loaded file: the copy doubled the load's peak RAM).

        Returns:
            The batch, as new tensors unless ``inplace``.
        """
        f, mask, attn_mask, *edep_layers = batch
        if dataset_already_normalised(f, self.pen.max_energy):
            return batch
        f = f if inplace else f.clone()
        s = _F(f)
        e_in, pid = s.in_energy[..., 0].clone(), s.in_pid   # MeV, before it becomes the scalar below
        s.in_energy.copy_(self.pen.to_scalar_e(s.in_energy))
        s.edep.copy_(self.norm_edep_token(s.edep, e_in, pid))
        if edep_layers:   # optional (B, n_layers) tensor; norm_edep is elementwise
            return f, mask, attn_mask, self.norm_edep(edep_layers[0])
        return f, mask, attn_mask

    # ---- E_dep: the log scale -----------------------------------------------------------------------------------

    def norm_edep(self, edep: Tensor) -> Tensor:
        """Converts E_dep from MeV to the log scale, held in [0, 1].

        ``u = log(edep / e_min) / log(max_energy / e_min)``. A deposit below ``e_min`` (zero included) gives 0.
        """
        if not self.edep_log_min: return norm_edep_linear(edep, self.pen.max_energy)   # legacy
        e_min = self.edep_log_min
        width = torch.log(torch.tensor(self.pen.max_energy / e_min)).item()
        u = torch.log((edep / e_min).clamp_min(1.0)) / width
        return u.clamp(0.0, 1.0)

    def denorm_edep(self, u: Tensor) -> Tensor:
        """Converts E_dep from the log scale to MeV, the inverse of ``norm_edep`` on [0, 1].

        ``edep = e_min exp(u log(max_energy / e_min))``. Every non-positive input (the ``overflow_delta`` sentinel)
        decodes to zero.
        """
        if not self.edep_log_min: return denorm_edep_linear(u, self.pen.max_energy)   # legacy
        e_min = self.edep_log_min
        width = torch.log(torch.tensor(self.pen.max_energy / e_min)).item()
        edep = e_min * torch.exp(u.clamp(0.0, 1.0) * width)
        return torch.where(u > 0, edep, torch.zeros_like(u))

    # ---- E_dep: the margin token --------------------------------------------------------------------------------

    def norm_edep_token(self, edep: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from MeV to the model scale: the margin token where ``edep_margin_mask``.

        ```
        z = log(edep) - log(cap - edep)          cap = e_in + margin(species)
        u = (z - z_lo) / (z_hi - z_lo)
        ```

        held in [0, 1]: ``cap`` is the edge ``u = 1``, and zero stays 0 (the overflow sentinel's condition). Every
        other event takes the log scale (``norm_edep``).

        Args:
            edep: the deposit [MeV].
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.

        Returns:
            The token, in the shape and dtype of ``edep``.
        """
        if not self.edep_margin:
            return self.norm_edep(edep)
        e = edep.double()
        cap, z_lo, z_hi = self._edep_margin_range(e_in.double(), pid)
        z = torch.log(e.clamp_min(_TINY)) - torch.log((cap - e).clamp_min(_TINY))
        u = ((z - z_lo) / (z_hi - z_lo)).clamp(0.0, 1.0)
        u = torch.where(e > 0, u, torch.zeros_like(u)).to(edep.dtype)
        return torch.where(self.edep_margin_mask(pid), u, self.norm_edep(edep))

    def denorm_edep_token(self, u: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from the model scale to MeV, the inverse of ``norm_edep_token``.

        ``edep = cap sigmoid(z_lo + u (z_hi - z_lo))`` on a margin token. A non-positive ``u`` (the sentinel) decodes to
        zero, ``u > 1`` to the cap.

        Args:
            u:    the token.
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.

        Returns:
            The deposit [MeV], in the shape and dtype of ``u``.
        """
        if not self.edep_margin:
            return self.denorm_edep(u)
        cap, z_lo, z_hi = self._edep_margin_range(e_in.double(), pid)
        edep = cap * torch.sigmoid(z_lo + u.double().clamp(0.0, 1.0) * (z_hi - z_lo))
        edep = torch.where(u > 0, edep, torch.zeros_like(edep)).to(u.dtype)
        return torch.where(self.edep_margin_mask(pid), edep, self.denorm_edep(u))

    def edep_margin_mask(self, pid: Tensor) -> Tensor:
        """Returns where the E_dep token is the margin token.

        Nowhere without ``model_conf.edep_margin``; with it for every incoming species, or only for those listed in
        ``model_conf.edep_margin_species`` (raw pdgids; the rest keep the log scale).
        """
        if not self.edep_margin:
            return torch.zeros_like(pid, dtype=torch.bool)
        mask = torch.full_like(pid, self.edep_margin_species is None, dtype=torch.bool)
        for p in self.edep_margin_species or ():   # python scalars: no host-to-device copy per call
            mask |= pid == p
        return mask

    def _edep_margin_range(self, e_in: Tensor, pid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the margin token's cap and log-odds range, ``(cap, z_lo, z_hi)``.

        ``cap = e_in + margin(species)``, ``z_lo = log(e_min / cap)``, ``z_hi = log(cap / 1e-3 MeV)``. A species
        without a margin of its own gets the widest.
        """
        margins = self.edep_excess_mev or HAND_SET_MARGIN_MEV   # legacy: a checkpoint from before edep_excess_mev
        margin = torch.full_like(e_in, max(margins.values()))
        for p, m in margins.items():
            margin = torch.where(pid == p, torch.full_like(e_in, m), margin)
        cap = e_in + margin
        z_lo = torch.log((self.edep_log_min or 1e-6) / cap)
        z_hi = torch.log(cap / _MISSING_MEV)
        return cap, z_lo, z_hi

    @staticmethod
    def edep_excess_max(f: Tensor) -> dict[int, float]:
        """Returns, per incoming pdgid, the largest ``edep - e_in`` [MeV] (>= 0) in a stored file: the margin.

        ``f`` holds MeV channels (before ``norm_e``). The result is meta.json's ``edep_excess_max_mev``, which becomes
        ``model_conf.edep_excess_mev``.
        """
        e_in, pid = _F(f).in_energy[..., 0].double(), _F(f).in_pid
        excess = _F(f).edep.reshape(e_in.shape).double() - e_in
        return {int(p): max(0.0, excess[pid == p].max().item()) for p in pid.unique().tolist()}
