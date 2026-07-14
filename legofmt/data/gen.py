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

n_items = 500        # one (energy, density) draw per item
n_events_per_item = 4000

pos = np.array([-50.0, 0.0, 0.0], dtype=np.float64)   # ignored with random_gun=True
mom = np.array([1.0, 0.0, 0.0], dtype=np.float64)

energy = np.array([300.0], dtype=np.float64)
energy_max = 300.0
energy_min = 10.0
density = np.linspace(0.5, 10.0, n_items)
# density = np.array([3.0], dtype=np.float64)
size = np.array([100.0], dtype=np.float64)
pdgids_in = np.array([-11, 11, 22], dtype=np.int32)

data = pyg4lego.run_simulation(
    n_events_per_item, pos, mom, energy,
    random_energy_emax=energy_max, 
    random_energy_emin=energy_min,
    random_gun=True,
    density=density,
    size=size,
    random_energy=True,
    SourceParticles=pdgids_in,
)
data = {grp: {k: torch.as_tensor(v) for k, v in d.items()} for grp, d in data.items()}

manifold = [{"name": "euclidean", "dim": 1 }, {"name": "sphere", "dim": 3 }, {"name": "sphere", "dim": 3 }]

config = {
    "cutoff_mev": energy_min,
    "max_energy": energy_max,
    "manifold": manifold,
    "proj_ray": True,
}

dataset = LEGODataset(
    data=data,
    prep=DataPrep(config),
    cutoff_mev=energy_min,
    min_particles=0,
    is_filtered=False,
    device="cpu",
)

out_dir = "<PATH>" # make this the path to the folder where this script lives
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
}

with open(f"{out_dir}/meta.json", "w") as f:
    json.dump(meta_dict, f, ensure_ascii=True, indent=4)