"""``DataPrep``: raw MeV particles -> the model's tokens, and E_dep to and from its scales.

A raw particle is ``[E_kin, p (3), x (3), pdgid]`` (MeV, mm). ``prep`` writes what a dataset file stores; ``norm_e``
puts the two channels that depend on ``max_energy`` on the model scale when the file is loaded, so a dataset is
generated without knowing ``max_energy``.

```
energy     incoming   e_in [MeV]  --norm_e-->  log(e_in / cutoff) / log(max_energy / cutoff)
           outgoing   1 - log(e_out / cutoff) / log(e_in / cutoff)
momentum              p / |p|
position              direction of x / cuboid_dim             incoming: x ray-traced onto the cube first

E_dep      layer scale    u = log(edep / e_min) / log(max_energy / e_min)        e_min = edep_log_min
           margin scale   z = logit(edep / cap) = log(edep) - log(cap - edep)    cap  = e_in + margin(species)
                          u = (z - z_lo) / (z_hi - z_lo)                         z_lo = log(e_min / cap)
                                                                                 z_hi = log(cap / 1e-3 MeV)
```

Every channel lands in [0, 1]. The E_dep token takes the margin scale (``model_conf.edep_margin``): the log-odds of
the deposited fraction of the cap. A small deposit keeps a log scale, and near the cap the channel resolves the
missing energy ``cap - edep``, so a contained shower's narrow peak below ``e_in`` becomes wide. ``margin(species)`` is
the training data's largest ``edep - e_in`` (``model_conf.edep_excess_mev``). The per-layer deposits, and the token of
an incoming species not in ``model_conf.edep_margin_species``, take the layer scale.

Superseded: the linear E_dep scale and hand-set margins (:mod:`legofmt.legacy.norm`), ``energy_kin: false`` and
``proj_ray: false`` (one line each in ``cc_trafo``).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.compat import dataset_already_normalised
from legofmt.data.struct import _F, cond_scalars, set_layout
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.legacy.norm import HAND_SET_MARGIN_MEV, denorm_edep_linear, norm_edep_linear
from legofmt.mod_comps.config import build_manifold

_MISSING_MEV = 1e-3   # the smallest missing energy cap - edep the margin scale resolves


class DataPrep:
    """Converts raw MeV particles to the model's tokens (``prep``, ``norm_e``), and E_dep to and from its scales.

    Args:
        config: a full config, or the flat dict of ``scripts/gen_data.py`` and ``MultModel`` (``MultModel`` gives no
            manifold: it only calls ``norm_e``).
    """

    def __init__(self, config: dict):
        config = config.get("config", config)
        if "model_conf" in config:   # a full config
            conf = config["model_conf"]
            cutoff_mev, self.proj_ray = config["dl_conf"]["lds_args"].get("cutoff_mev"), conf.get("proj_ray", True)
            self.manifold = build_manifold(conf["manifold"])
        else:                        # the flat dict
            conf = config
            cutoff_mev, self.proj_ray = conf.get("cutoff_mev"), conf.get("proj_ray")
            self.manifold = build_manifold(conf["manifold"]) if conf.get("manifold") else None
        self.pen = EnergyProjections(cutoff_mev=cutoff_mev, max_energy=conf.get("max_energy"))
        self.ppa = CubeTrace(conf.get("cuboid_dim"))
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
        set_layout(conf.get("cond_scalars", ("Density",)))   # process-global

    def __call__(self, batch: tuple) -> tuple:
        return self.prep(batch)

    # ---- raw particles -> stored tokens -------------------------------------------------------------------------

    @torch.no_grad()
    def prep(self, batch: tuple) -> tuple:
        """Converts a raw batch to the padded model layout, as a dataset file stores it.

        The two ``max_energy``-dependent channels (incoming energy, E_dep) stay in MeV; ``norm_e`` converts them at
        load time.

        Args:
            batch: ``(particles, m, am, per_event)`` as ``GetLEGOData`` returns it: (B, 1 + n_out, 8) raw particles,
                the incoming one first, their masks, and the per-event scalars.

        Returns:
            ``(f, m, am)``, with the per-layer E_dep [MeV] as a 4th tensor when ``per_event`` carries it.
        """
        particles, m, am, per_event = batch
        e_kin = particles[..., 0:1] if self.energy_kin else None
        tokens = torch.cat((self.cc_trafo(particles[..., 1:7], e_kin=e_kin), particles[..., 7:]), dim=-1)
        return self.format_add((tokens, m, am, per_event))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        """Converts MeV momenta and positions to (energy, unit momentum direction, unit position direction).

        Args:
            cc:    (B, 1 + n_out, 6), momentum and position per particle, the incoming one first; NaN is padding.
            e_kin: (B, 1 + n_out, 1), Geant4's recorded kinetic energy [MeV].

        Returns:
            (B, 1 + n_out, 7), on the manifold; the incoming energy stays in MeV.
        """
        mom, pos = cc.nan_to_num(1).split(3, -1)
        e_mev = (e_kin.nan_to_num(1) if e_kin is not None
                 else mom.norm(dim=-1, keepdim=True))   # energy_kin false (legacy): |p|, saturates for hadrons

        log_e   = torch.log(e_mev.clamp_min(1e-8) / self.pen.cutoff)
        e_out   = 1 - (log_e[:, 1:] / log_e[:, :1].clamp_min(1e-6)).clamp(0, 1)
        energy  = torch.cat((e_mev[:, :1], e_out), dim=1)
        mom_dir = F.normalize(mom, dim=-1)
        if self.proj_ray:   # the incoming position, traced along its momentum onto the cube; false: legacy
            pos = pos.select_scatter(self.ppa(torch.cat((mom_dir[:, 0], pos[:, 0]), dim=-1))[..., 3:], dim=1, index=0)
        pos_dir = F.normalize(pos / self.ppa.cuboid_dim.to(pos), dim=-1)
        return self.manifold.projx(torch.cat((energy, mom_dir, pos_dir), dim=-1))

    @torch.no_grad()
    def format_add(self, batch: tuple) -> tuple:
        """Prepends the prefix tokens: the conditioning scalars and the generated E_dep (the layout of ``struct.py``).

        Args:
            batch: ``(tokens, m, am, per_event)``: (B, 1 + n_out, 8) particle tokens, their masks, and the per-event
                scalars by name (``E_dep``, the ``cond_scalars``, optionally ``E_dep_layers``).

        Returns:
            ``(f, m, am)``, E_dep the only prefix token with ``m = 1``; the (B, n_layers) ``E_dep_layers`` [MeV] as a
            4th tensor.
        """
        tokens, m, am, per_event = batch
        first, *rest = cond_scalars()
        names = (first, "E_dep", *rest)
        prefix = tokens.new_tensor([0, 1, 1, 1, 1, 1, 1, 0]).repeat(len(tokens), len(names), 1)
        prefix[..., 0] = torch.stack([per_event[n].reshape(len(tokens)) for n in names], dim=1)

        f = torch.cat((prefix, tokens), dim=1).nan_to_num()
        m = F.pad(m, (len(names), 0))
        m[:, 1] = 1   # E_dep, the one generated prefix token
        am = F.pad(am, (len(names), 0), value=True)
        return (f, m, am, per_event["E_dep_layers"]) if "E_dep_layers" in per_event else (f, m, am)

    # ---- stored tokens -> the model scale -----------------------------------------------------------------------

    @torch.no_grad()
    def norm_e(self, batch: tuple, inplace: bool = False) -> tuple:
        """Converts the two channels that depend on ``max_energy`` from MeV to the model scale.

        The incoming energy becomes ``log(e_in / cutoff) / log(max_energy / cutoff)``, the E_dep token its scale
        (``norm_edep_margin``), the per-layer deposits theirs (``norm_edep_layer``). A dataset written before the
        energy split is passed through; see ``compat.dataset_already_normalised``.

        Args:
            batch:   ``(f, m, am[, edep_layers])``.
            inplace: write into ``f`` (a freshly loaded file: the copy doubled the load's peak RAM).

        Returns:
            The batch, as new tensors unless ``inplace``.
        """
        f, m, am, *edep_layers = batch
        if dataset_already_normalised(f, self.pen.max_energy):
            return batch
        s = _F(f if inplace else f.clone())
        s.edep.copy_(self.norm_edep_margin(s.edep, s.in_energy[..., 0], s.in_pid))   # reads e_in in MeV: first
        s.in_energy.copy_(self.pen.to_scalar_e(s.in_energy))
        return (s.full, m, am, *map(self.norm_edep_layer, edep_layers))

    # ---- E_dep: the layer scale ---------------------------------------------------------------------------------

    def norm_edep_layer(self, edep: Tensor) -> Tensor:
        """Converts E_dep from MeV to the layer scale, held in [0, 1]: a deposit below ``e_min`` (zero too) gives 0."""
        if not self.edep_log_min: return norm_edep_linear(edep, self.pen.max_energy)   # legacy
        e_min = self.edep_log_min
        width = torch.log(torch.tensor(self.pen.max_energy / e_min)).item()
        return (torch.log((edep / e_min).clamp_min(1.0)) / width).clamp_max(1.0)

    def denorm_edep_layer(self, u: Tensor) -> Tensor:
        """Converts E_dep from the layer scale to MeV: ``e_min exp(u log(max_energy / e_min))``, 0 where ``u <= 0``."""
        if not self.edep_log_min: return denorm_edep_linear(u, self.pen.max_energy)   # legacy
        e_min = self.edep_log_min
        width = torch.log(torch.tensor(self.pen.max_energy / e_min)).item()
        return torch.where(u > 0, e_min * torch.exp(u.clamp(0.0, 1.0) * width), 0.0)

    # ---- E_dep: the margin scale (the token) --------------------------------------------------------------------

    def norm_edep_margin(self, edep: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from MeV to the margin scale, held in [0, 1]: ``cap`` gives 1, zero gives 0.

        Where ``edep_margin_mask`` is false, the token takes the layer scale.

        Args:
            edep: the deposit [MeV].
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.

        Returns:
            The token, in the shape and dtype of ``edep``.
        """
        if not self.edep_margin: return self.norm_edep_layer(edep)
        cap, z_lo, z_hi = self._edep_margin_range(e_in.double(), pid)
        z = torch.logit((edep.double() / cap).clamp(0.0, 1.0))   # -inf at zero, +inf from the cap on
        u = ((z - z_lo) / (z_hi - z_lo)).clamp(0.0, 1.0).to(edep.dtype)
        return torch.where(self.edep_margin_mask(pid), u, self.norm_edep_layer(edep))

    def denorm_edep_margin(self, u: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from the margin scale to MeV: ``cap sigmoid(z_lo + u (z_hi - z_lo))``.

        ``u <= 0`` decodes to zero, ``u > 1`` to the cap. Where ``edep_margin_mask`` is false, the token is on the
        layer scale.

        Args:
            u:    the token.
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.

        Returns:
            The deposit [MeV], in the shape and dtype of ``u``.
        """
        if not self.edep_margin: return self.denorm_edep_layer(u)
        cap, z_lo, z_hi = self._edep_margin_range(e_in.double(), pid)
        edep = cap * torch.sigmoid(z_lo + u.double().clamp(0.0, 1.0) * (z_hi - z_lo))
        edep = torch.where(u > 0, edep, 0.0).to(u.dtype)
        return torch.where(self.edep_margin_mask(pid), edep, self.denorm_edep_layer(u))

    def edep_margin_mask(self, pid: Tensor) -> Tensor:
        """Returns where the E_dep token is on the margin scale: every species, or those in ``edep_margin_species``."""
        if not self.edep_margin: return torch.zeros_like(pid, dtype=torch.bool)
        if self.edep_margin_species is None: return torch.ones_like(pid, dtype=torch.bool)
        return torch.isin(pid, pid.new_tensor(self.edep_margin_species))

    def _edep_margin_range(self, e_in: Tensor, pid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(cap, z_lo, z_hi)``; a species without a margin of its own gets the widest."""
        margins = self.edep_excess_mev or HAND_SET_MARGIN_MEV   # legacy: a checkpoint from before edep_excess_mev
        own = pid[..., None] == pid.new_tensor(list(margins))   # (..., n_species), one-hot or all false
        mev = e_in.new_tensor(list(margins.values()))
        cap = e_in + torch.where(own.any(-1), (own * mev).sum(-1), mev.max())
        return cap, torch.log((self.edep_log_min or 1e-6) / cap), torch.log(cap / _MISSING_MEV)

    @staticmethod
    def edep_excess_max(f: Tensor) -> dict[int, float]:
        """Returns, per incoming pdgid, the largest ``edep - e_in`` [MeV] (>= 0) in a stored file: the margin.

        ``f`` holds MeV channels (before ``norm_e``). The result is meta.json's ``edep_excess_max_mev``, which becomes
        ``model_conf.edep_excess_mev``.
        """
        s = _F(f)
        e_in = s.in_energy.reshape(-1).double()
        species, idx = s.in_pid.reshape(-1).unique(return_inverse=True)
        excess = e_in.new_zeros(len(species)).scatter_reduce(0, idx, s.edep.reshape(-1).double() - e_in, "amax")
        return dict(zip(species.long().tolist(), excess.tolist()))   # the zeros: an excess >= 0
