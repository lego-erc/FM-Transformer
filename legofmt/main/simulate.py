"""
Unified simulation API: ``simulate_fm`` and ``simulate_g4`` accept the same
physical-space arguments and both return data in the format produced by
``GenerateOut.gen_model_w_g4_args``::

    {
        "per_event":    {"E_dep": (N,), "Density": (N,)},
        "per_particle": {"Incoming": (N, 1, 8), "Outgoing": (N, max_p, 8)},
        "per_voxel":    {"E_dep": (N, 0, 4)},
    }

Particle features: ``[energy, px, py, pz, pdgid, x, y, z]``.
Outgoing particles are in model space; incoming is in physical space.
``E_dep`` is normalised by incoming energy (fraction).
Inactive outgoing particles are zero-filled.

Serialization
-------------
Use :func:`save_result` / :func:`load_result` to persist simulation output.
The file stores all tensors (moved to CPU) alongside the physics configuration
that produced them, so downstream code (metrics, plotting) is self-contained::

    save_result("g4_150mev.pt", result, source="g4",
                pos=pos, mom=mom, energy=energy, density=density,
                size=size, pdgids=pdgids)
    data = load_result("g4_150mev.pt")
    data["result"]   # the dict of tensors
    data["config"]   # the physics config
"""

import torch
import pyg4lego
from pathlib import Path

from ..geometry.energy_proj import EnergyProjections
from ..geometry.path_sample_mult import ProductManifold
from flow_matching.utils.manifolds import Euclidean, Sphere

from .generate import GenerateOut


def _physical_incoming(n, pos, mom, energy, pdgids):
    """Build physical-space incoming particle: [energy, px, py, pz, pdgid, x, y, z]."""
    e = energy.float().expand(n, 1)
    p = mom.float().view(1, 3).expand(n, 3)
    pid = pdgids.float().view(1, 1).expand(n, 1)
    x = pos.float().view(1, 3).expand(n, 3)
    return torch.cat([e, p, pid, x], dim=-1).unsqueeze(1)  # (N, 1, 8)


def make_simulate_fm(gen: GenerateOut):
    """Return a ``simulate_fm`` callable with the same signature as ``simulate_g4``.

    The returned closure captures ``gen`` so the call signature is::

        simulate_fm(n, pos, mom, energy, density, size, pdgids)
    """

    def simulate_fm(
        n: int,
        pos: torch.Tensor,
        mom: torch.Tensor,
        energy: torch.Tensor,
        density: torch.Tensor,
        size: torch.Tensor,
        pdgids: torch.Tensor,
    ):
        mom_dir = mom.float() / mom.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if mom_dir.dim() == 1:
            mom_dir = mom_dir.unsqueeze(0)

        pos_f = pos.float()
        if pos_f.dim() == 1:
            pos_f = pos_f.unsqueeze(0)
        # Model expects position on unit sphere, not physical mm
        pos_unit = pos_f / pos_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            result = gen.gen_model_w_g4_args(
                n, pos_unit, mom_dir, energy.float(),
                density.float(), size.float(), pdgids,
            )
        result["per_particle"]["Incoming"] = _physical_incoming(
            n, pos, mom, energy, pdgids,
        ).to(result["per_particle"]["Outgoing"].device)
        return result

    return simulate_fm


def make_simulate_fm_base(gen: GenerateOut):
    """Return a ``simulate_fm_base`` callable: multiplicity model + base distribution (t=0).

    Same API as ``simulate_fm`` but skips the ODE integration, returning the
    flow's base distribution. Useful as a baseline to measure how much the
    flow matching ODE improves over the prior.
    """

    def simulate_fm_base(
        n: int,
        pos: torch.Tensor,
        mom: torch.Tensor,
        energy: torch.Tensor,
        density: torch.Tensor,
        size: torch.Tensor,
        pdgids: torch.Tensor,
    ):
        mom_dir = mom.float() / mom.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if mom_dir.dim() == 1:
            mom_dir = mom_dir.unsqueeze(0)

        pos_f = pos.float()
        if pos_f.dim() == 1:
            pos_f = pos_f.unsqueeze(0)
        pos_unit = pos_f / pos_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            result = gen.gen_model_w_g4_args(
                n, pos_unit, mom_dir, energy.float(),
                density.float(), size.float(), pdgids,
                return_base=True,
            )
        result["per_particle"]["Incoming"] = _physical_incoming(
            n, pos, mom, energy, pdgids,
        ).to(result["per_particle"]["Outgoing"].device)
        return result

    return simulate_fm_base


def simulate_g4(
    n: int,
    pos: torch.Tensor,
    mom: torch.Tensor,
    energy: torch.Tensor,
    density: torch.Tensor,
    size: torch.Tensor,
    pdgids: torch.Tensor,
    cutoff_mev: float = 10.0,
):
    """Run Geant4 via pyg4lego, forward-transform to model space.

    Outgoing particles with physical energy below ``cutoff_mev`` are zeroed
    out (matching the training-data cut applied to the FM model).

    Returns exactly the same dict structure as ``simulate_fm`` /
    ``GenerateOut.gen_model_w_g4_args``.
    """
    result = pyg4lego.run_simulation(
        n,
        pos.double(), mom.double(), energy.double(),
        density=density.double(), size=size.double(),
        n_voxels=1, random_gun=False, random_energy=False,
        SourceParticles=pdgids.int(),
    )

    # ------------------------------------------------------------------
    # G4 raw layout: [energy, px, py, pz, x, y, z, pdgid]
    # ------------------------------------------------------------------
    inc_raw = result["per_particle"]["Incoming"].float()   # (N, 1, 8)
    out_raw = result["per_particle"]["Outgoing"].float()   # (N, max_p, 8)
    e_dep   = result["per_event"]["E_dep"].float()         # (N,)

    # ------------------------------------------------------------------
    # Forward transform physical → model space
    # ------------------------------------------------------------------
    # Build 6D [px, py, pz, x, y, z] — G4 position lives at indices 4:7
    inc_cc = torch.cat([inc_raw[..., 1:4], inc_raw[..., 4:7]], dim=-1)
    out_cc = torch.cat([out_raw[..., 1:4], out_raw[..., 4:7]], dim=-1)

    all_cc = torch.cat([inc_cc, out_cc], dim=1)  # (N, 1+max_p, 6)
    en_proj = EnergyProjections("in_frac_log")
    manifold = ProductManifold([Euclidean(), Sphere()], (3, 3))
    all_cc = manifold.projx(en_proj(all_cc))

    inc_model = all_cc[:, :1]   # (N, 1, 6)
    out_model = all_cc[:, 1:]   # (N, max_p, 6)

    # ------------------------------------------------------------------
    # Reassemble into FM format: [energy, px, py, pz, pdgid, x, y, z]
    # ------------------------------------------------------------------
    def _to_particle(cc_6d, pdgid_vals):
        """Same logic as gen_model_w_g4_args._to_particle."""
        mom_, pos_ = cc_6d.split(3, -1)
        e = mom_.norm(dim=-1, keepdim=True)
        return torch.cat([e, mom_, pdgid_vals.unsqueeze(-1).float(), pos_], dim=-1)

    incoming = _physical_incoming(n, pos, mom, energy, pdgids)

    # Outgoing pdgids from G4 (index 7)
    out_pdgid = out_raw[..., 7]  # (N, max_p)
    outgoing = _to_particle(out_model, out_pdgid)

    # Zero out inactive particles: empty rows OR below energy cutoff
    # (same convention as gen_model_w_g4_args where inactive → zeros)
    invalid = (out_raw.abs().sum(-1) == 0) | (out_raw[..., 0] < cutoff_mev)
    outgoing[invalid] = 0.0

    return {
        "per_event": {
            "E_dep": e_dep / energy.float(),
            "Density": density.float().expand(n),
        },
        "per_particle": {
            "Incoming": incoming,
            "Outgoing": outgoing,
        },
        "per_voxel": {
            "E_dep": torch.empty(n, 0, 4),
        },
    }


# ======================================================================
# Serialization
# ======================================================================

def _to_cpu(result):
    """Recursively move all tensors in a result dict to CPU."""
    out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = _to_cpu(v)
        elif isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        else:
            out[k] = v
    return out


def save_dataset(
    path,
    *,
    fm_result: dict,
    g4_result: dict,
    pos: torch.Tensor,
    mom: torch.Tensor,
    energy: torch.Tensor,
    density: torch.Tensor,
    size: torch.Tensor,
    pdgids: torch.Tensor,
):
    """Save paired FM + G4 results for a single input condition.

    Parameters
    ----------
    path : str or Path
        Output file path (conventionally ``.pt``).
    fm_result, g4_result : dict
        Outputs of ``simulate_fm`` and ``simulate_g4``.
    pos, mom, energy, density, size, pdgids :
        The shared physics configuration.
    """
    torch.save(
        {
            "fm": _to_cpu(fm_result),
            "g4": _to_cpu(g4_result),
            "config": {
                "n_fm": fm_result["per_event"]["E_dep"].shape[0],
                "n_g4": g4_result["per_event"]["E_dep"].shape[0],
                "pos": pos.cpu(),
                "mom": mom.cpu(),
                "energy": energy.cpu(),
                "density": density.cpu(),
                "size": size.cpu(),
                "pdgids": pdgids.cpu(),
            },
        },
        path,
    )


def load_dataset(path):
    """Load a paired FM + G4 dataset.

    Returns
    -------
    dict with keys ``"fm"``, ``"g4"`` (tensor dicts), ``"config"``,
    and optionally ``"fm_base"``.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def add_to_dataset(path, key, result):
    """Add a new result (e.g. ``"fm_base"``) to an existing dataset file."""
    ds = torch.load(path, map_location="cpu", weights_only=False)
    ds[key] = _to_cpu(result)
    torch.save(ds, path)
