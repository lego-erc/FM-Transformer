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

The two E_dep scales are ``LayerScale`` and ``MarginScale`` below, as formulas; ``DataPrep`` picks one per token and
applies it.

Superseded: the linear E_dep scale and hand-set margins (:mod:`legofmt.legacy.norm`), ``energy_kin: false`` and
``proj_ray: false`` (one line each in ``cc_trafo``).
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.compat import dataset_already_normalised, edep_margin_keys
from legofmt.data.struct import _F, cond_scalars, set_layout, species_values
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.legacy.norm import HAND_SET_MARGIN_MEV, denorm_edep_linear, norm_edep_linear
from legofmt.config import build_manifold

_MISSING_MEV = 1e-3   # the smallest missing energy cap - edep the margin scale resolves


def token_energy(e: Tensor, cutoff: float) -> Tensor:
    """Returns the particles' energy channel, (B, 1 + n_out, 1) with the incoming particle first.

    ```
    incoming   e_in                                           [MeV]; norm_e puts it on the model scale at load time
    outgoing   1 - log(e_out / cutoff) / log(e_in / cutoff)   held in [0, 1]
    ```
    """
    log_e = torch.log(e.clamp(min=1e-8) / cutoff)   # every particle's, in one call
    log_e_in, log_e_out = log_e[:, :1], log_e[:, 1:]
    e_out = 1 - (log_e_out / log_e_in.clamp(min=1e-6)).clamp(0, 1)
    return torch.cat((e[:, :1], e_out), dim=1)


class LayerScale(NamedTuple):
    """E_dep's layer scale, held in [0, 1].

    ```
    norm     u    = log(edep / e_min) / width,   width = log(max_energy / e_min)    0 below e_min, 1 from max_energy on
    denorm   edep = e_min exp(u width)                                               0 where u <= 0
    ```
    """
    e_min: float
    width: float

    def norm(self, edep):
        e_min, width = self
        return (torch.log((edep / e_min).clamp(min=1.0)) / width).clamp(max=1.0)

    def denorm(self, u):
        e_min, width = self
        return torch.where(u > 0, e_min * torch.exp(u.clamp(0.0, 1.0) * width), 0.0)


class MarginScale(NamedTuple):
    """The E_dep token's margin scale, per event, held in [0, 1].

    ```
    norm     z    = logit(edep / cap),   u = (z - z_lo) / (z_hi - z_lo)       0 at zero, 1 from the cap on
    denorm   edep = cap sigmoid(z_lo + u (z_hi - z_lo))                       0 where u <= 0
    ```
    """
    cap:  Tensor
    z_lo: Tensor
    z_hi: Tensor

    def norm(self, edep):
        cap, z_lo, z_hi = self
        z = torch.logit((edep / cap).clamp(0.0, 1.0))
        return ((z - z_lo) / (z_hi - z_lo)).clamp(0.0, 1.0)

    def denorm(self, u):
        cap, z_lo, z_hi = self
        edep = cap * torch.sigmoid(z_lo + u.clamp(0.0, 1.0) * (z_hi - z_lo))
        return torch.where(u > 0, edep, 0.0)


class DataPrep:
    """Converts raw MeV particles to the model's tokens (``prep``, ``norm_e``), and E_dep to and from its scales.

    Args:
        config: a full config, or the flat dict of ``scripts/gen_data.py`` and ``MultModel`` (``MultModel`` gives no
            manifold: it only calls ``norm_e``).
    """

    def __init__(self, config: dict):
        conf, cutoff_mev, self.proj_ray, self.manifold = _read_config(config)
        self.pen             = EnergyProjections(cutoff_mev=cutoff_mev, max_energy=conf.get("max_energy"))
        self.ppa             = CubeTrace(conf.get("cuboid_dim"))
        self.energy_kin      = conf.get("energy_kin", True)
        self.edep_log_min    = conf.get("edep_log_min")
        self.edep_margin, self.edep_margin_species = edep_margin_keys(conf)
        self.edep_excess_mev = conf.get("edep_excess_mev")
        _check_margins(self.edep_margin, self.edep_margin_species, self.edep_excess_mev)
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
        e_kin, cc, pid = particles.tensor_split((1, 7), dim=-1)   # [E_kin, p (3), x (3), pdgid]
        e_kin = e_kin if self.energy_kin else None                 # false: legacy, |p|
        tokens = torch.cat((self.cc_trafo(cc, e_kin=e_kin), pid), dim=-1)
        return self.format_add((tokens, m, am, per_event))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        """Converts MeV momenta and positions to (energy, unit momentum direction, unit position direction).

        ```
        energy     incoming   e_in [MeV]                                      token_energy
                   outgoing   1 - log(e_out / cutoff) / log(e_in / cutoff)
        momentum              p / |p|
        position              x / cuboid_dim, normalised       the incoming x first ray-traced along p onto the cube
        ```

        Args:
            cc:    (B, 1 + n_out, 6), momentum and position per particle, the incoming one first; NaN is padding.
            e_kin: (B, 1 + n_out, 1), Geant4's recorded kinetic energy [MeV].

        Returns:
            (B, 1 + n_out, 7), on the manifold; the incoming energy stays in MeV.
        """
        e, p, x = _unpadded(cc, e_kin)
        energy  = token_energy(e, self.pen.cutoff)
        mom_dir = F.normalize(p, dim=-1)
        x       = self._ray_traced(mom_dir, x)
        pos_dir = F.normalize(x / self.ppa.cuboid_dim.to(x), dim=-1)
        return self.manifold.projx(torch.cat((energy, mom_dir, pos_dir), dim=-1))

    @torch.no_grad()
    def format_add(self, batch: tuple) -> tuple:
        """Prepends the prefix tokens: the conditioning scalars and the generated E_dep (the layout of ``struct.py``).

        ```
        prefix token   [value, 1, 1, 1, 1, 1, 1, 0]       for cond_scalars[0], E_dep, cond_scalars[1:], in this order
        m              1 on E_dep, 0 on the other ones    E_dep is the one generated prefix token
        am             True on every prefix token
        ```

        Args:
            batch: ``(tokens, m, am, per_event)``: (B, 1 + n_out, 8) particle tokens, their masks, and the per-event
                scalars by name (``E_dep``, the ``cond_scalars``, optionally ``E_dep_layers``).

        Returns:
            ``(f, m, am)``, E_dep the only prefix token with ``m = 1``; the (B, n_layers) ``E_dep_layers`` [MeV] as a
            4th tensor.
        """
        tokens, m, am, per_event = batch
        first, *rest = cond_scalars()
        values = [per_event[name] for name in (first, "E_dep", *rest)]
        f  = torch.cat((_prefix_tokens(values, like=tokens), tokens), dim=1).nan_to_num()
        m  = F.pad(m, (len(values), 0))
        m[:, 1] = 1   # E_dep
        am = F.pad(am, (len(values), 0), value=True)
        return (f, m, am, per_event["E_dep_layers"]) if "E_dep_layers" in per_event else (f, m, am)

    # ---- stored tokens -> the model scale -----------------------------------------------------------------------

    @torch.no_grad()
    def norm_e(self, batch: tuple, inplace: bool = False) -> tuple:
        """Converts the two channels that depend on ``max_energy`` from MeV to the model scale.

        ```
        E_dep token   norm_edep_margin(edep, e_in, pid)                  first: it reads e_in in MeV
        e_in          log(e_in / cutoff) / log(max_energy / cutoff)      held in [0, 1]
        edep_layers   norm_edep_layer(edep_layers)
        ```

        A dataset written before the energy split is passed through; see ``compat.dataset_already_normalised``.

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

    # ---- E_dep: the layer scale (LayerScale) ---------------------------------------------------------------------

    def norm_edep_layer(self, edep: Tensor) -> Tensor:
        """Converts E_dep from MeV to the layer scale: a deposit below ``e_min`` (zero too) gives 0."""
        if not self.edep_log_min: return norm_edep_linear(edep, self.pen.max_energy)   # legacy
        return self._layer_scale().norm(edep)

    def denorm_edep_layer(self, u: Tensor) -> Tensor:
        """Converts E_dep from the layer scale to MeV: 0 where ``u <= 0``."""
        if not self.edep_log_min: return denorm_edep_linear(u, self.pen.max_energy)   # legacy
        return self._layer_scale().denorm(u)

    # ---- E_dep: the margin scale (MarginScale), the token's ------------------------------------------------------

    def norm_edep_margin(self, edep: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from MeV to the margin scale, in float64: ``cap`` gives 1, zero gives 0.

        Where ``edep_margin_mask`` is false, the token takes the layer scale.

        Args:
            edep: the deposit [MeV].
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.

        Returns:
            The token, in the shape and dtype of ``edep``.
        """
        if not self.edep_margin: return self.norm_edep_layer(edep)
        u = self._margin_scale(e_in.double(), pid).norm(edep.double()).to(edep.dtype)
        return torch.where(self.edep_margin_mask(pid), u, self.norm_edep_layer(edep))

    def denorm_edep_margin(self, u: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from the margin scale to MeV, in float64.

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
        edep = self._margin_scale(e_in.double(), pid).denorm(u.double()).to(u.dtype)
        return torch.where(self.edep_margin_mask(pid), edep, self.denorm_edep_layer(u))

    def edep_margin_mask(self, pid: Tensor) -> Tensor:
        """Returns where the E_dep token is on the margin scale: every species, or those in ``edep_margin_species``."""
        if not self.edep_margin: return torch.zeros_like(pid, dtype=torch.bool)
        if self.edep_margin_species is None: return torch.ones_like(pid, dtype=torch.bool)
        return torch.isin(pid, pid.new_tensor(self.edep_margin_species))

    @staticmethod
    def edep_excess_max(f: Tensor) -> dict[int, float]:
        """Returns, per incoming pdgid, the largest ``edep - e_in`` [MeV] (>= 0) in a stored file: the margin.

        ```
        margin(species) = max(0, max over the species' events of edep - e_in)
        ```

        ``f`` holds MeV channels (before ``norm_e``). The result is meta.json's ``edep_excess_max_mev``, which becomes
        ``model_conf.edep_excess_mev``.
        """
        s = _F(f)
        e_in = s.in_energy.flatten().double()
        excess = s.edep.flatten().double() - e_in
        species, idx = s.in_pid.flatten().unique(return_inverse=True)
        margin = e_in.new_zeros(len(species)).scatter_reduce(0, idx, excess, "amax")   # per species; the zeros: >= 0
        return dict(zip(species.long().tolist(), margin.tolist()))

    # ---- plumbing ------------------------------------------------------------------------------------------------

    def _ray_traced(self, mom_dir: Tensor, x: Tensor) -> Tensor:
        """Returns ``x`` with the incoming particle's position traced along its momentum onto the cube.

        ``proj_ray: false`` (legacy) leaves it as it is.
        """
        if not self.proj_ray: return x   # legacy
        incoming = torch.cat((mom_dir[:, 0], x[:, 0]), dim=-1)
        return x.select_scatter(self.ppa(incoming)[..., 3:], dim=1, index=0)

    def _layer_scale(self) -> LayerScale:
        """Returns the layer scale; ``width`` is a float32 log, as stored scales were made."""
        e_min = self.edep_log_min
        return LayerScale(e_min, torch.log(torch.tensor(self.pen.max_energy / e_min)).item())

    def _margin_scale(self, e_in: Tensor, pid: Tensor) -> MarginScale:
        """Returns each event's margin scale.

        ```
        cap  = e_in + margin(species)        a species without a margin of its own gets the widest
        z_lo = log(e_min / cap)
        z_hi = log(cap / 1e-3 MeV)
        ```
        """
        margins = self.edep_excess_mev or HAND_SET_MARGIN_MEV   # legacy: a checkpoint from before edep_excess_mev
        mev, known = species_values(margins, pid, e_in)
        cap = e_in + torch.where(known, mev, max(margins.values()))
        return MarginScale(cap, torch.log((self.edep_log_min or 1e-6) / cap), torch.log(cap / _MISSING_MEV))


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _read_config(config: dict) -> tuple:
    """Returns ``(conf, cutoff_mev, proj_ray, manifold)`` of a full config or of the flat dict.

    The flat dict is ``scripts/gen_data.py``'s and ``MultModel``'s; ``MultModel``'s has no manifold, it only calls
    ``norm_e``.
    """
    config = config.get("config", config)
    if "model_conf" in config:   # a full config
        conf = config["model_conf"]
        return (conf, config["dl_conf"]["lds_args"].get("cutoff_mev"), conf.get("proj_ray", True),
                build_manifold(conf["manifold"]))
    manifold = build_manifold(config["manifold"]) if config.get("manifold") else None
    return config, config.get("cutoff_mev"), config.get("proj_ray"), manifold


def _unpadded(cc: Tensor, e_kin: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """Returns ``(e, p, x)`` [MeV, mm] with the NaN padding set to 1, so that it stays finite through the formulas.

    ``e`` is ``e_kin``, or ``|p|`` without it (``energy_kin: false``, legacy: it saturates for hadrons).
    """
    p, x = cc.nan_to_num(1).split(3, dim=-1)
    e = e_kin.nan_to_num(1) if e_kin is not None else p.norm(dim=-1, keepdim=True)   # legacy
    return e, p, x


def _prefix_tokens(values: list, like: Tensor) -> Tensor:
    """Returns the (B, len(values), 8) prefix tokens ``[value, 1, 1, 1, 1, 1, 1, 0]``, in ``like``'s dtype, device."""
    prefix = like.new_tensor([0, 1, 1, 1, 1, 1, 1, 0]).repeat(len(like), len(values), 1)
    prefix[..., 0] = torch.stack([v.flatten() for v in values], dim=1)
    return prefix


def _check_margins(margin, species, excess_mev) -> None:
    """Raises if a species on the margin scale has no margin.

    Raises:
        ValueError: an ``edep_margin_species`` entry is missing from ``edep_excess_mev``.
    """
    if margin and excess_mev is not None:
        missing = set(species or ()) - set(excess_mev)
        if missing:
            raise ValueError(f"edep_margin_species {sorted(missing)} have no entry in model_conf.edep_excess_mev: "
                             "no such incoming species in the training data, so no cap")
