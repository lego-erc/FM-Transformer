"""``DataPrep``: raw MeV particles -> the model's tokens, and the energies to and from their scales.

A raw particle is ``[E_kin, p (3), x (3), pdgid]`` (MeV, mm). ``prep`` converts it to what a dataset file stores, every
energy in MeV. ``norm_e`` puts the energies on the model scale when the file is loaded, so a dataset holds no
parameter of a model (``max_energy``, the margins).

```
energy     incoming   e_in [MeV]   --norm_e-->  log(e_in / cutoff) / log(max_energy / cutoff)
           outgoing   e_out [MeV]  --norm_e-->  e = 1 - u of the margin scale             lo = cutoff
momentum              p / |p|
position              direction of x / cuboid_dim             incoming: x ray-traced onto the cube first

E_dep      layer scale    u = log(edep / e_min) / log(max_energy / e_min)        e_min = edep_log_min
           margin scale   u of the margin scale                                  lo = e_min

margin     z = logit(x / cap) = log(x) - log(cap - x)       cap  = e_in + margin
           u = (z - z_lo) / (z_hi - z_lo)                    z_lo = log(lo / cap),   z_hi = log(cap / 1e-3 MeV)
```

Every channel but the directions lies in [0, 1]. The margin scale is the log-odds of a fraction of the cap: a small
value keeps a log scale, and near the cap the scale resolves the missing energy ``cap - x``.

- The outgoing energy (``model_conf.out_margin``): ``margin(pid_in, pid_out)`` is the training data's largest
  ``e_out - e_in`` per pair of species (``model_conf.out_excess_mev``), 0 for a pair of one species. So a surviving
  particle's energy loss spreads over the middle of [0, 1], and a decay or capture product keeps an energy above
  ``e_in``. Turned round, ``e = 1 - u``, it keeps the legacy scale's ends: 0 at the cap, 1 at the cutoff.
- The E_dep token (``model_conf.edep_margin``): ``margin(species)`` is the training data's largest ``edep - e_in``
  per incoming species (``model_conf.edep_excess_mev``). A contained shower's narrow peak below ``e_in`` becomes wide.
  The per-layer deposits, and the token of an incoming species not in ``model_conf.edep_margin_species``, take the
  layer scale.

``LayerScale`` and ``MarginScale`` below are the two scales as formulas; ``DataPrep`` picks one per token and applies
it.

Legacy (:mod:`legofmt.legacy.norm`): the linear E_dep scale, the hand-set margins, and the log scale of the outgoing
energy, which a model without ``out_margin`` takes and every dataset from before stores; ``energy_kin: false`` and
``proj_ray: false`` (lines marked ``# legacy`` here).
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.compat import dataset_already_normalised, edep_margin_keys
from legofmt.data.struct import _F, cond_scalars, set_layout, species_values, vocab_index
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.legacy.norm import (HAND_SET_MARGIN_MEV, denorm_edep_linear, denorm_out_energy_log, norm_edep_linear,
                                 norm_out_energy_log)
from legofmt.config import build_manifold

_MISSING_MEV = 1e-3   # the smallest missing energy cap - edep the margin scale resolves


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
    """The margin scale of the E_dep token and the outgoing energy, per particle, held in [0, 1].

    ```
    norm     z = logit(x / cap),   u = (z - z_lo) / (z_hi - z_lo)     0 at x = lo cap / (cap + lo), 1 at cap - ~1e-3 MeV
    denorm   x = cap sigmoid(z_lo + u (z_hi - z_lo))
    ```
    """
    cap:  Tensor
    z_lo: Tensor
    z_hi: Tensor

    def norm(self, x):
        cap, z_lo, z_hi = self
        z = torch.logit((x / cap).clamp(0.0, 1.0))
        return ((z - z_lo) / (z_hi - z_lo)).clamp(0.0, 1.0)

    def denorm(self, u):
        cap, z_lo, z_hi = self
        return cap * torch.sigmoid(z_lo + u.clamp(0.0, 1.0) * (z_hi - z_lo))


class DataPrep:
    """Raw MeV particles -> the model's tokens (``prep``, ``norm_e``), and E_dep to and from its scales.

    Args:
        config: a full config (or a checkpoint that holds one under ``config``), or the flat dict of
            ``scripts/gen_data.py`` and ``MultModel`` (``MultModel`` gives no manifold: it only calls ``norm_e``).
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
        self.out_margin      = conf.get("out_margin", False)
        self.out_excess_mev  = conf.get("out_excess_mev")
        self._out_margins    = _pair_table(self.out_excess_mev) if self.out_margin else None
        set_layout(conf.get("cond_scalars", ("Density",)))   # process-global

    def __call__(self, batch: tuple) -> tuple:
        return self.prep(batch)

    # ---- raw particles -> stored tokens -------------------------------------------------------------------------

    @torch.no_grad()
    def prep(self, batch: tuple) -> tuple:
        """Converts a raw batch to the padded model layout, as a dataset file stores it.

        The values that depend on ``max_energy`` (the incoming energy, E_dep, the per-layer E_dep) stay in MeV;
        ``norm_e`` converts them at load time.

        Args:
            batch: ``(particles, m, am, per_event)`` as ``GetLEGOData`` returns it: (B, 1 + n_out, 8) raw particles,
                the incoming one first, their masks, and the per-event scalars.

        Returns:
            ``(f, m, am)``, with the per-layer E_dep [MeV] as a 4th tensor when ``per_event`` carries it.
        """
        particles, m, am, per_event = batch
        e_kin, cc, pid = particles.tensor_split((1, 7), dim=-1)   # [E_kin, p (3), x (3), pdgid]
        e_kin = e_kin if self.energy_kin else None                 # None (legacy): the energy is |p|
        tokens = torch.cat((self.cc_trafo(cc, e_kin=e_kin), pid), dim=-1)
        return self.format_add((tokens, m, am, per_event))

    @torch.no_grad()
    def cc_trafo(self, cc: Tensor, e_kin: Tensor | None = None) -> Tensor:
        """Converts momenta [MeV] and positions [mm] to (energy, unit momentum direction, unit position direction).

        ```
        energy     [MeV], as it is                              norm_e puts it on the model scale at load time
        momentum   p / |p|
        position   x / cuboid_dim, normalised                   the incoming x first ray-traced along p onto the cube
        ```

        Args:
            cc:    (B, 1 + n_out, 6), momentum and position per particle, the incoming one first; NaN is padding.
            e_kin: (B, 1 + n_out, 1), Geant4's recorded kinetic energy [MeV]; None (legacy): ``|p|`` is the energy.

        Returns:
            (B, 1 + n_out, 7), on the manifold, with the energies in MeV.
        """
        e, p, x = _unpadded(cc, e_kin)
        mom_dir = F.normalize(p, dim=-1)
        x       = self._ray_traced(mom_dir, x)
        pos_dir = F.normalize(x / self.ppa.cuboid_dim.to(x), dim=-1)
        return self.manifold.projx(torch.cat((e, mom_dir, pos_dir), dim=-1))

    @torch.no_grad()
    def format_add(self, batch: tuple) -> tuple:
        """Prepends the prefix tokens: the conditioning scalars and the generated E_dep (the layout of ``struct.py``).

        ```
        prefix token   [value, 1, 1, 1, 1, 1, 1, 0]       for cond_scalars[0], E_dep, cond_scalars[1:], in this order
        m              1 on E_dep, 0 on the other ones    E_dep is the one generated prefix token
        am             True on every prefix token
        ```

        A NaN that is left (the pdgid of a padding token) becomes 0.

        Args:
            batch: ``(tokens, m, am, per_event)``: (B, 1 + n_out, 8) particle tokens, their masks, and the per-event
                scalars by name (``E_dep``, the ``cond_scalars``, optionally ``E_dep_layers``).

        Returns:
            ``(f, m, am)`` with ``f`` (B, n_prefix + 1 + n_out, 8); the (B, n_layers) ``E_dep_layers`` [MeV] as a 4th
            tensor when ``per_event`` carries it.
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
    def norm_e(self, batch: tuple, inplace: bool = False, out_mev: bool = True) -> tuple:
        """Converts the energies from MeV to the model scale.

        ```
        outgoing energy   norm_out_energy(e_out, e_in, pid_in, pid_out)      first: these two read e_in in MeV
        E_dep token       norm_edep_margin(edep, e_in, pid)
        e_in              log(e_in / cutoff) / log(max_energy / cutoff)      held in [0, 1]
        edep_layers       norm_edep_layer(edep_layers)
        ```

        The padding keeps its values. A legacy dataset that already stores the incoming energy and E_dep on the model
        scale is returned as it is (``compat.dataset_already_normalised``).

        Args:
            batch:   ``(f, m, am[, edep_layers])``; the masks are passed on unread.
            inplace: write into ``f`` instead of a copy (for a freshly loaded file: the copy doubles the peak RAM of
                the load).
            out_mev: the outgoing energy is in MeV, as ``prep`` writes it. False (legacy): a file from before, which
                stores it on the legacy log scale; it stays as it is.

        Returns:
            ``(f, m, am[, edep_layers])`` on the model scale.

        Raises:
            ValueError: ``out_margin`` on a dataset from before (``out_mev`` false).
        """
        f, m, am, *edep_layers = batch
        if self.out_margin and not out_mev:
            raise ValueError("model_conf.out_margin needs the outgoing energy in MeV, but this dataset stores it on "
                             "the legacy log scale, clamped at E_in: regenerate it with scripts/gen_data.py")
        if dataset_already_normalised(f, self.pen.max_energy):
            return batch
        s = _F(f if inplace else f.clone())
        if out_mev:   # reads e_in in MeV: first
            e_out = self.norm_out_energy(s.out_energy, s.in_p[..., 0], s.in_p[..., -1], s.out_pid)
            s.out_energy.copy_(torch.where(s.out_pid != 0, e_out, s.out_energy))
        s.edep.copy_(self.norm_edep_margin(s.edep, s.in_energy[..., 0], s.in_pid))   # reads e_in in MeV: first
        s.in_energy.copy_(self.pen.to_scalar_e(s.in_energy))
        return (s.full, m, am, *map(self.norm_edep_layer, edep_layers))

    # ---- the outgoing energy: the margin scale, turned round -----------------------------------------------------

    def norm_out_energy(self, e_out: Tensor, e_in: Tensor, pid_in: Tensor, pid_out: Tensor) -> Tensor:
        """Converts the outgoing energy from MeV to the model scale, computed in float64: the cap gives 0.

        ``e = 1 - u`` of the margin scale (``_out_scale``). Without ``out_margin`` (legacy): the log scale that every
        dataset from before stores.

        Args:
            e_out:   the outgoing energy [MeV].
            e_in:    the incoming kinetic energy [MeV], broadcast against ``e_out``.
            pid_in:  the incoming raw pdgid, broadcast against ``e_out``.
            pid_out: the outgoing raw pdgid, in the shape of ``e_out``.

        Returns:
            The energy channel, in the shape and dtype of ``e_out``.
        """
        if not self.out_margin: return norm_out_energy_log(e_out, e_in, self.pen.cutoff)   # legacy
        return (1 - self._out_scale(e_in.double(), pid_in, pid_out).norm(e_out.double())).to(e_out.dtype)

    def denorm_out_energy(self, e: Tensor, e_in: Tensor, pid_in: Tensor, pid_out: Tensor) -> Tensor:
        """Converts the outgoing energy from the model scale to MeV, computed in float64: ``norm_out_energy`` inverted.

        ``e`` is held in [0, 1]: 0 decodes to 1e-3 MeV below the cap, 1 to ``cap cutoff / (cap + cutoff)``. The
        arguments are those of ``norm_out_energy``, with ``e`` in the place of ``e_out``.

        Returns:
            The energy [MeV], in the shape and dtype of ``e``.
        """
        if not self.out_margin: return denorm_out_energy_log(e, e_in, self.pen.cutoff)   # legacy
        return self._out_scale(e_in.double(), pid_in, pid_out).denorm(1 - e.double()).to(e.dtype)

    @staticmethod
    def out_excess_max(f: Tensor) -> dict[int, dict[int, float]]:
        """Returns, per pair of incoming and outgoing pdgid, the largest ``e_out - e_in`` [MeV] (>= 0) in a stored file.

        ```
        margin(pid_in, pid_out) = max(0, max over the pair's outgoing particles of e_out - e_in)
        ```

        ``f`` is a stored file's features, every energy in MeV (before ``norm_e``). The padding (pdgid 0) and the pairs
        of one species, whose margin is 0, do not count. The result is meta.json's ``out_excess_max_mev``, which
        becomes ``model_conf.out_excess_mev``.
        """
        s = _F(f)
        pid_in = s.in_p[..., -1].expand_as(s.out_pid)
        pair   = (s.out_pid != 0) & (s.out_pid != pid_in)
        excess = (s.out_energy.double() - s.in_energy.double())[pair]
        pairs, idx = torch.stack((pid_in[pair], s.out_pid[pair]), -1).long().unique(dim=0, return_inverse=True)
        # the largest excess per pair; the zeros it starts from keep it >= 0
        margin = excess.new_zeros(len(pairs)).scatter_reduce(0, idx, excess, "amax")
        table: dict[int, dict[int, float]] = {}
        for (p, q), m in zip(pairs.tolist(), margin.tolist()):
            table.setdefault(p, {})[q] = m
        return table

    # ---- E_dep: the layer scale (LayerScale) ---------------------------------------------------------------------

    def norm_edep_layer(self, edep: Tensor) -> Tensor:
        """Converts E_dep from MeV to the layer scale: a deposit below ``edep_log_min`` (zero too) gives 0."""
        if not self.edep_log_min: return norm_edep_linear(edep, self.pen.max_energy)   # legacy
        return self._layer_scale().norm(edep)

    def denorm_edep_layer(self, u: Tensor) -> Tensor:
        """Converts E_dep from the layer scale to MeV: 0 where ``u <= 0``."""
        if not self.edep_log_min: return denorm_edep_linear(u, self.pen.max_energy)   # legacy
        return self._layer_scale().denorm(u)

    # ---- E_dep: the margin scale (MarginScale), the token's ------------------------------------------------------

    def norm_edep_margin(self, edep: Tensor, e_in: Tensor, pid: Tensor) -> Tensor:
        """Converts the E_dep token from MeV to the margin scale, computed in float64: ``cap`` gives 1, zero gives 0.

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
        """Converts the E_dep token from the margin scale to MeV, computed in float64.

        ``u <= 0`` decodes to zero, ``u >= 1`` to 1e-3 MeV below the cap. Where ``edep_margin_mask`` is false, the token
        is on the layer scale.

        Args:
            u:    the token.
            e_in: the incoming kinetic energy [MeV], same shape.
            pid:  the incoming raw pdgid, same shape.

        Returns:
            The deposit [MeV], in the shape and dtype of ``u``.
        """
        if not self.edep_margin: return self.denorm_edep_layer(u)
        edep = torch.where(u > 0, self._margin_scale(e_in.double(), pid).denorm(u.double()), 0.0).to(u.dtype)
        return torch.where(self.edep_margin_mask(pid), edep, self.denorm_edep_layer(u))

    def edep_margin_mask(self, pid: Tensor) -> Tensor:
        """Returns where the E_dep token is on the margin scale, by the incoming pdgid ``pid``.

        Nowhere without ``edep_margin``; else for every species, or for those in ``edep_margin_species``.
        """
        if not self.edep_margin: return torch.zeros_like(pid, dtype=torch.bool)
        if self.edep_margin_species is None: return torch.ones_like(pid, dtype=torch.bool)
        return torch.isin(pid, pid.new_tensor(self.edep_margin_species))

    @staticmethod
    def edep_excess_max(f: Tensor) -> dict[int, float]:
        """Returns, per incoming pdgid, the largest ``edep - e_in`` [MeV] (>= 0) in a stored file: the margin.

        ```
        margin(species) = max(0, max over the species' events of edep - e_in)
        ```

        ``f`` is a stored file's features, with the incoming energy and E_dep in MeV (before ``norm_e``). The result is
        meta.json's ``edep_excess_max_mev``, which becomes ``model_conf.edep_excess_mev``.
        """
        s = _F(f)
        e_in = s.in_energy.flatten().double()
        excess = s.edep.flatten().double() - e_in
        species, idx = s.in_pid.flatten().unique(return_inverse=True)
        # the largest excess per species; the zeros it starts from keep it >= 0
        margin = e_in.new_zeros(len(species)).scatter_reduce(0, idx, excess, "amax")
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
        """Returns the layer scale; ``width`` is a float32 log, so the scale stays bit-identical to the one in use."""
        e_min = self.edep_log_min
        return LayerScale(e_min, torch.log(torch.tensor(self.pen.max_energy / e_min)).item())

    def _margin_scale(self, e_in: Tensor, pid: Tensor) -> MarginScale:
        """Returns each event's margin scale.

        ```
        cap  = e_in + margin(species)        a species without a margin of its own gets the widest
        z_lo = log(e_min / cap)              e_min = edep_log_min, 1e-6 MeV without one
        z_hi = log(cap / 1e-3 MeV)
        ```
        """
        margins = self.edep_excess_mev or HAND_SET_MARGIN_MEV   # legacy: a checkpoint without ``edep_excess_mev``
        mev, known = species_values(margins, pid, e_in)
        cap = e_in + torch.where(known, mev, max(margins.values()))
        return MarginScale(cap, torch.log((self.edep_log_min or 1e-6) / cap), torch.log(cap / _MISSING_MEV))

    def _out_scale(self, e_in: Tensor, pid_in: Tensor, pid_out: Tensor) -> MarginScale:
        """Returns each outgoing particle's margin scale.

        ```
        cap  = e_in + margin(pid_in, pid_out)
        z_lo = log(cutoff / cap)
        z_hi = log(cap / 1e-3 MeV)
        ```

        ``margin`` is ``out_excess_mev[pid_in][pid_out]``, 0 for a pair of one species. A pair the table does not hold
        takes the widest margin of its incoming species, an incoming species it does not hold the widest of all.
        """
        pid_in, pid_out = pid_in.long(), pid_out.long()
        vin, vout, margins = (t.to(pid_out.device) for t in self._out_margins)
        i, known_in  = vocab_index(vin, pid_in)
        j, known_out = vocab_index(vout, pid_out)
        margin = margins[torch.where(known_in, i, len(vin)), torch.where(known_out, j, len(vout))]
        cap = e_in + torch.where(pid_in == pid_out, 0.0, margin)
        return MarginScale(cap, torch.log(self.pen.cutoff / cap), torch.log(cap / _MISSING_MEV))


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _read_config(config: dict) -> tuple:
    """Returns ``(conf, cutoff_mev, proj_ray, manifold)`` of a full config or of the flat dict.

    A checkpoint is read through its ``config``. The flat dict is ``scripts/gen_data.py``'s and ``MultModel``'s;
    ``MultModel``'s has no manifold, it only calls ``norm_e``.
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

    ``e`` is ``e_kin``, or ``|p|`` without it (``energy_kin: false``, legacy: a hadron's ``|p|`` exceeds its kinetic
    energy and saturates the scale).
    """
    p, x = cc.nan_to_num(1).split(3, dim=-1)
    e = e_kin.nan_to_num(1) if e_kin is not None else p.norm(dim=-1, keepdim=True)   # legacy
    return e, p, x


def _prefix_tokens(values: list, like: Tensor) -> Tensor:
    """Returns the (B, len(values), 8) prefix tokens ``[value, 1, 1, 1, 1, 1, 1, 0]``, in ``like``'s dtype, device."""
    prefix = like.new_tensor([0, 1, 1, 1, 1, 1, 1, 0]).repeat(len(like), len(values), 1)
    prefix[..., 0] = torch.stack([v.flatten() for v in values], dim=1)
    return prefix


def _pair_table(excess: dict | None) -> tuple[Tensor, Tensor, Tensor]:
    """Returns ``out_excess_mev`` as ``(pid_in, pid_out, margins)``: two sorted vocabularies and a dense table.

    ```
    margins[i, j]         = excess[pid_in[i]][pid_out[j]]       the pair's margin [MeV]
                          = max of excess[pid_in[i]]             a pair the table does not hold, and the last column
    margins[len(pid_in)]  = the largest margin of all            the last row: an incoming species it does not hold
    ```

    Raises:
        ValueError: no table: ``model_conf.out_margin`` without ``model_conf.out_excess_mev``.
    """
    if excess is None:
        raise ValueError("model_conf.out_margin needs model_conf.out_excess_mev, the margins per pair of species "
                         "(meta.json's out_excess_max_mev): regenerate the dataset with scripts/gen_data.py")
    pid_in, pid_out = sorted(excess), sorted({q for row in excess.values() for q in row})
    widest = {p: max(excess[p].values(), default=0.0) for p in pid_in}
    rows = [[excess[p].get(q, widest[p]) for q in pid_out] + [widest[p]] for p in pid_in]
    rows.append([max(widest.values(), default=0.0)] * (len(pid_out) + 1))
    return torch.tensor(pid_in), torch.tensor(pid_out), torch.tensor(rows, dtype=torch.float64)


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
