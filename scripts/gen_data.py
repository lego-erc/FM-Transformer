"""Prepare raw shower data for training.

The events are simulated in chunks, each its own process (Geant4 is seeded from
std::random_device, so chunks are independent and may run in parallel), and merged once into
``data_packed.pt`` (the prepped tensors without their padding, see ``legofmt.data.packed``;
``--padded`` writes the padded ``data_prepped.pt`` instead) and ``meta.json`` (the metadata the
fresh-training config path reads). OUT is the first CLI argument, or ``$LEGO_DATA_DIR`` if none
is given:

    python gen_data.py [OUT] [--padded]          # all chunks, one after the other, then merge none
    python gen_data.py OUT chunk I               # I = 0 .. n_chunks-1 -> OUT/chunks/chunk_I.pt (cut and compacted)
    python gen_data.py OUT merge stats           # outgoing multiplicity, and what each cap would drop
    python gen_data.py OUT merge MAX_OUT [--padded]   # drop events with > MAX_OUT outgoing (none: keep all), prep, write
"""

import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pyg4lego

from legofmt.data.dataloaders import GetLEGOData
from legofmt.data.packed import PackedData
from legofmt.data.prep import DataPrep

n_items = 500        # density points swept per material
n_events_per_item = 4000
n_chunks = 10

pos = np.array([-50.0, 0.0, 0.0])   # ignored with random_gun=True
mom = np.array([1.0, 0.0, 0.0])

energy_max = 300.0
energy_min = 10.0
energy = np.array([energy_max])   # ignored with random_energy=True
pdgids_in = np.array([-11, 11, 22], dtype=np.int32)

materials = [               # (Z, A, cube edge [mm])
    (18.0, 39.95, 100.0),   # argon
    (29.0, 63.546, 10.0),   # copper
]
density_space = np.linspace(0.5, 10.0, n_items)

density = np.tile(density_space, len(materials))
atomic_number, mass_number, size = np.repeat(np.array(materials).T, n_items, axis=1)

padded = "--padded" in sys.argv[1:]   # the padded data_prepped.pt, for legofmt versions without packed files
args = [a for a in sys.argv[1:] if a != "--padded"]
out_dir = args[0] if args else os.environ.get("LEGO_DATA_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
mode, arg = (args[1:3] + [None, None])[:2]

if mode is None:
    # all chunks, one after the other, each in its own process (one Geant4 run manager per process)
    for i in range(n_chunks):
        print(f"chunk {i + 1}/{n_chunks}", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), out_dir, "chunk", str(i)], check=True)
    print(f"all {n_chunks} chunks written to {out_dir}/chunks, merging", flush=True)
    arg = "none"   # then merge, keeping every event

if mode == "chunk":
    rows = np.array_split(np.arange(len(density)), n_chunks)[int(arg)]
    data = pyg4lego.run_simulation(
        n_events_per_item, pos, mom, energy,
        random_energy_emax=energy_max,
        random_energy_emin=energy_min,
        random_gun=True,
        density=density[rows],
        atomic_number=atomic_number[rows],
        mass_number=mass_number[rows],
        size=size[rows],
        random_energy=True,
        SourceParticles=pdgids_in,
    )
    data = {grp: {k: torch.as_tensor(v) for k, v in d.items()} for grp, d in data.items()}
    if "per_slab" in data:   # per-layer E_dep (slab geometry): stored as a 4th tensor next to (f, m, am)
        data["per_event"]["E_dep_layers"] = data["per_slab"]["E_dep"]
    os.makedirs(f"{out_dir}/chunks", exist_ok=True)
    torch.save(GetLEGOData(cutoff_mev=energy_min, min_particles=0)(data), f"{out_dir}/chunks/chunk_{arg}.pt")
    sys.exit()

# Everything from here on is the merge.
torch.set_num_threads(8)   # torch defaults to every core; on a busy box the extra threads only spin-wait
manifold = [{"name": "euclidean", "dim": 1 }, {"name": "sphere", "dim": 3 }, {"name": "sphere", "dim": 3 }]

# Per-event conditioning scalars, in slot order. Must match the per_event keys
# emitted by pyg4lego, and be identical in the training config; otherwise the
# extra scalars are silently dropped / the slot layout mismatches.
cond_scalars = ["Density", "Z", "A", "Size"]

# Edge lengths [Lx, Ly, Lz] of a cuboid target (only their ratios are used); None = cube.
# Positions are stored as the direction of x / cuboid_dim, so the model needs the same value.
cuboid_dim = None

# max_energy is not used by DataPrep.prep -- the stored file keeps the
# max_energy-dependent channels in MeV and DataPrep.norm_e converts them at
# load time. It goes into meta.json only as the default for a fresh model.
config = {
    "cutoff_mev": energy_min,
    "manifold": manifold,
    "proj_ray": True,
    "cond_scalars": cond_scalars,
    "cuboid_dim": cuboid_dim,
}
prep = DataPrep(config)

# One chunk in memory at a time (a 20M-event set went past 400 GB at once). DataPrep is per event, so prepping
# chunk by chunk (packed, or into preallocated padded outputs) gives what one call over all events would.
# mmap for the n_out pass only: it reads just the attention masks. The merge pass reads everything, and on
# GPFS faulting a whole chunk in through mmap takes ~5x as long as a plain load.
paths = [f"{out_dir}/chunks/chunk_{i}.pt" for i in range(n_chunks)]
n_out = [torch.load(p, weights_only=False, mmap=True)[2].sum(1) - 1 for p in paths]   # outgoing per event; row 0 is the incoming
sizes = [len(n) for n in n_out]
n_out = torch.cat(n_out)
if arg == "stats":
    print(f"{len(n_out)} events, outgoing mean {n_out.float().mean():.2f}, max {int(n_out.max())}")
    for cap in (32, 64, 96, 128, 160, 192, 256):
        print(f"MAX_OUT {cap:4d} drops {int((n_out > cap).sum()):8d} events ({(n_out > cap).float().mean():.5f})")
    sys.exit()
cap = int(n_out.max()) if arg == "none" else int(arg)
keep = n_out <= cap
L = 1 + int(n_out[keep].max())
out, i, chunks, excess = None, 0, [], {}
for path, kp in zip(paths, keep.split(sizes)):
    pf, pm, pam, padd = torch.load(path, weights_only=False)
    w = L - min(L, pf.shape[1])
    parts = prep((F.pad(pf[kp, :L], (0, 0, 0, w), value=float("nan")),
                  F.pad(pm[kp, :L], (0, w)),
                  F.pad(pam[kp, :L], (0, w)),
                  {k: v[kp] for k, v in padd.items()}))
    del pf, pm, pam, padd
    for p, e in DataPrep.edep_excess_max(parts[0]).items():   # per incoming pdgid: the E_dep cap's margin
        excess[p] = max(excess.get(p, 0.0), e)
    n_edep_layers = parts[3].shape[1] if len(parts) > 3 else None
    if padded:   # (f, m, am[, E_dep_layers])
        if out is None:
            out = [t.new_empty((int(keep.sum()),) + t.shape[1:]) for t in parts]
        for o, t in zip(out, parts):
            o[i:i + len(t)] = t
        i += len(parts[0])
    else:
        chunks.append(PackedData.from_padded(*parts[:3], n_prefix=len(cond_scalars) + 1,
                                             edep_layers=parts[3] if len(parts) > 3 else None))
    del parts

if padded:
    torch.save(tuple(out), f"{out_dir}/data_prepped.pt")
    ntokens, pids = out[0].shape[1], out[0][..., -1]
else:
    packed = PackedData.concat(chunks)
    del chunks
    torch.save(packed.to_dict(), f"{out_dir}/data_packed.pt")
    ntokens, pids = packed.L, packed.rows[:, -1]

pdgids = pids.flatten().nan_to_num().unique()
pdgids = pdgids[(0 < pdgids.abs()) & (pdgids.abs() < 100000000)].tolist() # filter pdgids

meta_dict = {
    "ntokens": ntokens,
    "particles": pdgids,
    "particles_in": pdgids_in.tolist(),
    "max_energy": energy_max,
    "cutoff_mev": energy_min,
    "cond_scalars": cond_scalars,
    "energy_kin": True,
    "cuboid_dim": cuboid_dim,
    "n_edep_layers": n_edep_layers,
    "edep_excess_max_mev": dict(sorted(excess.items())),
}
if not padded:
    meta_dict.update(format="packed", n_events=len(packed))

with open(f"{out_dir}/meta.json", "w") as f:
    json.dump(meta_dict, f, ensure_ascii=True, indent=4)
