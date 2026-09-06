"""
This script will create a data_prepped.pt pickle file with prepared data as well as a meta.json file containing metadata.
These files will (/should) be placed in the same folder as this script."""

import json
import os

import numpy as np
import torch
import pyg4lego

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.prep import DataPrep

rng = np.random.default_rng(0)

n_items = 500        # density points swept per material
n_events_per_item = 4000

pos = np.array([-50.0, 0.0, 0.0], dtype=np.float64)   # ignored with random_gun=True
mom = np.array([1.0, 0.0, 0.0], dtype=np.float64)

energy = np.array([300.0], dtype=np.float64)
energy_max = 300.0
energy_min = 10.0
pdgids_in = np.array([-11, 11, 22], dtype=np.int32)

materials = [
    (18.0, 39.95, 100.0),   # argon
    (29.0, 63.546, 10.0),   # copper
]
density_space = np.linspace(0.5, 10.0, n_items)

density       = np.tile(density_space, len(materials))
atomic_number = np.repeat(np.array([z for z, _, _ in materials], dtype=np.float64), n_items)
mass_number   = np.repeat(np.array([a for _, a, _ in materials], dtype=np.float64), n_items)
size          = np.repeat(np.array([s for _, _, s in materials], dtype=np.float64), n_items)

data = pyg4lego.run_simulation(
    n_events_per_item, pos, mom, energy,
    random_energy_emax=energy_max,
    random_energy_emin=energy_min,
    random_gun=True,
    density=density,
    atomic_number=atomic_number,
    mass_number=mass_number,
    size=size,
    random_energy=True,
    SourceParticles=pdgids_in,
)
data = {grp: {k: torch.as_tensor(v) for k, v in d.items()} for grp, d in data.items()}

manifold = [{"name": "euclidean", "dim": 1 }, {"name": "sphere", "dim": 3 }, {"name": "sphere", "dim": 3 }]

# Per-event conditioning scalars, in slot order. Must match the per_event keys
# emitted by pyg4lego, and be identical in the training config; otherwise the
# extra scalars are silently dropped / the slot layout mismatches.
cond_scalars = ["Density", "Z", "A", "Size"]

# max_energy is not used by DataPrep.prep -- the stored file keeps the
# max_energy-dependent channels in MeV and DataPrep.norm_e converts them at
# load time. It goes into meta.json only as the default for a fresh model.
config = {
    "cutoff_mev": energy_min,
    "manifold": manifold,
    "proj_ray": True,
    "cond_scalars": cond_scalars,
}

dataset = LEGODataset(
    data=data,
    prep=DataPrep(config),
    cutoff_mev=energy_min,
    min_particles=0,
    is_filtered=False,
    device="cpu",
)

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

d = dataset.data
torch.save((d.f.full, d.m.full, d.am.full), f"{out_dir}/data_prepped.pt")

ntokens = d.f.full.shape[1]
pdgids = d.f.full[..., -1].flatten().nan_to_num().unique()
pdgids = pdgids[(0 < pdgids.abs()) & (pdgids.abs() < 1000000000)].tolist() # filter pdgids

meta_dict = {
    "ntokens": ntokens,
    "particles": pdgids,
    "particles_in": pdgids_in.tolist(),
    "max_energy": energy_max,
    "cutoff_mev": energy_min,
    "cond_scalars": cond_scalars,
}

with open(f"{out_dir}/meta.json", "w") as f:
    json.dump(meta_dict, f, ensure_ascii=True, indent=4)