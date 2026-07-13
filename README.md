# legofmt

Riemannian continuous flow-matching transformer for generating outgoing particles
given an incoming particle and a material density. Targets calorimeter-style
shower data (LEGO/Geant4) where each event has one incoming particle and a
variable number of outgoing particles labelled by PDG-id.

Generation is **two-stage**:

1. **Multiplicity model** (`MultModel`, autoregressive `x-transformers` decoder)
   predicts how many outgoing particles of each PDG-id the event contains.
2. **Flow-matching model** (`LEGOLtng` wrapping `CFMTrafo_x`, an `x-transformers`
   encoder) integrates an ODE on `Euclidean(3) × Sphere(2)` to produce per-particle
   `(momentum, surface-position)` for that many outgoing slots.

Both models are `lightning.LightningModule`s configured by a single nested
`config` dict.

---

## Quickstart — generate an event

Point the two paths at your local checkpoints and run:

```python
import torch
from legofmt.main.generate import GenerateOut

FLOW_CKPT = "PATH_TO_CHECKPOINTS/flow_ckpt.pt"   # flow-matching checkpoint: {"state_dict", "config"}
MULT_CKPT = "PATH_TO_CHECKPOINTS/mult_ckpt.pt"   # multiplicity checkpoint:  {"state_dict", "config"}
device = "cuda" if torch.cuda.is_available() else "cpu"

gen = GenerateOut(FLOW_CKPT, MULT_CKPT, device=device)

# One incoming particle; n samples are drawn from it.
n       = 1                                  # number of samples              total events will be n * B
pos     = torch.tensor([[0.0, 0.0, -50.0]])  # entry position (on surface)    (B, 3) or (1, 3)
mom     = torch.tensor([[0.0, 0.0, 1.0]])    # direction (auto-normalised)    (B, 3) or (1, 3)
energy  = torch.tensor([300.0])              # incoming energy, MeV           (B,) or (1,)
density = torch.tensor([3.0])                # material density               (B,) or (1,)
size    = torch.tensor([100.0])              # required, but unused
pdgids  = torch.tensor([11])                 # a PDG-id, e^- in this case;    (B,) or (1,)

with torch.no_grad():
    out = gen.gen_model_w_g4_args(n, pos, mom, energy, density, size, pdgids)

# out["per_particle"]["Outgoing"] : [B*n, max_seq_l-3, 8]
# out["per_particle"]["Incoming"] : [B*n, 1, 8]   layout: [density, px,py,pz, x,y,z, pdgid]
# out["per_event"]                : {"E_dep", "Density"}
# out["per_voxel"]                : {"E_dep": empty}
print(out["per_particle"]["Outgoing"].shape)
```

Each input has batch size `1` or `B`; with `B > 1` events, `n` samples are drawn
per event, so the output has `B*n` rows. `mom` is a **direction** — it is
normalised internally and scaled by `energy` — and `pos` is ray-traced onto the
conditioning cube for you. (This differs from the raw `gen(cond)` path below,
where the momentum must already be energy-scaled.)

---

## Workflow

```
Path A — ad-hoc (raw dict in memory):
    raw event dict ──► GetLEGOData ──► DataPrep ──► LEGODataset(dict, prep=...)
    (per_event +       (mom cutoff,    (manifold projx,
     per_particle)      NaN/sort)       ray-trace, en. proj)

Path B — pre-prepped on disk (used by every training entry point):
    folder/  or  .pt  ───────────────► LEGODataset(path)
    (data_prepped.pt +                  (loads the (target, mask, attn_mask)
     meta.json, already DataPrep'd)      3-tuple directly — no prep step)

Both paths yield a DataStruct stream:

    DataStruct ──► LEGOLtng  /  MultModel ──► trainer.fit
                                                   │
                                                   ▼
                                  torch.save({"state_dict", "config"})
                                                   │
                                                   ▼
                                  GenerateOut(flow_ckpt, mult_ckpt)
                                    · multiplicity sampling
                                    · ODE solve on product manifold
                                    · returns (sols, mask, attn_mask)
```

`LEGODataset` dispatches on its `data` argument: a folder or `.pt` path loads a
pre-prepped 3-tuple (path B); a raw dict requires `prep=DataPrep(config)` and
runs path A. Training entry points use path B exclusively
(`scripts/train.py` for reference).

### Training

```python
from legofmt.main.modules import LEGOLtng
import lightning as ltng

model   = LEGOLtng(config)                    # see config schema below
trainer = ltng.Trainer(max_epochs=10, accelerator="gpu", devices=[0,1,2,3],
                      strategy="ddp", precision=32)
trainer.fit(model)

# `model.model` is a ProjectModel wrapping the CFMTrafo_x vector field.
# If you torch.compile()d it, the original is at `model.model._orig_mod`.
vf_sd = (model.model._orig_mod.vf if hasattr(model.model, "_orig_mod")
         else model.model.vf).state_dict()
torch.save({"state_dict": vf_sd, "config": config}, "flow.pt")
```

The same checkpoint dict (`{"state_dict": ..., "config": ...}`) is what
`LEGOLtng(config)` expects when re-loading: pass the loaded dict in as `config`
and the constructor pulls both keys out. `MultModel` follows the same pattern.
A reference training entry point is in `scripts/train.py` (4-GPU DDP, Comet
logger, Muon optimizer with warmup-cosine).

### Generation — lower-level API

The Geant4-shaped entry point `gen_model_w_g4_args(...)` is covered in
[Quickstart](#quickstart--generate-an-event). Underneath it sits the raw `cond`
API (and the `couple_in_out_pdgids` constructor flag):

```python
from legofmt.main.generate import GenerateOut

gen = GenerateOut("flow.pt", "mult.pt", device="cuda",
                  couple_in_out_pdgids=False)  # if True, restrict outgoing
                                               # pdg-ids to the incoming set

# Raw call: cond is [B, 8] = [density, px,py,pz, x,y,z, pdgid_raw]
sols, mask, attn_mask = gen(cond)
# sols : [B, ntokens, 8]  =  [density, px,py,pz, x,y,z, pdgid_raw]
# mask / attn_mask : [B, ntokens]
```

---

## Data structure

### On disk

Each prepared dataset is a folder containing:

- `data_prepped.pt` — a 3-tuple `(target, mask, attn_mask)` matching `DataStruct.__init__`.
- `meta.json` — `{ "ntokens": int, "particles": [pdgid, ...], "particles_in": [pdgid, ...] }`.

`ntokens` is `2 + 1 + max_outgoing` (two scalar rows, one incoming, the rest outgoing).
`particles` is the sorted set of outgoing PDG-ids; `particles_in` is the set of
incoming PDG-ids the multiplicity model knows about.

### In memory (`DataStruct`)

`DataStruct(f, m, am)` wraps three tensors. `N = ntokens`.

| Field | Shape | Meaning |
|---|---|---|
| `f` (features) | `[B, N, 8]` | Per-particle features, see layout below. |
| `m` (loss mask) | `[B, N]` int | `1` where the slot is a random variable the flow must produce; `0` where it is a condition or pad. |
| `am` (attn mask) | `[B, N]` bool | `True` for valid slots (transformer attention mask). |

Row layout along `N`:

```
row 0       : non_p[0]   – per-event scalar row 1 (density)         mask=0  attn=1
row 1       : non_p[1]   – per-event scalar row 2 (E_dep / E_in)    mask=1  attn=1
row 2       : in_p       – incoming particle (condition)            mask=0  attn=1
rows 3..N-1 : out_p      – outgoing particles (RVs, padded)         mask=1  attn=1/0
```

Column layout along the last dim of `f` (8 columns total):

| Col | Particle rows (`in_p`, `out_p`) | Non-particle rows (`non_p`) |
|---|---|---|
| `0` | original scalar from raw data (e.g. energy `E`); **not consumed by the model** | `1` |
| `1` | `px` of momentum (or its `in_frac` / `log` transform after `DataPrep`) | row 0: **density**; row 1: **E_dep / E_in** |
| `2` | `py` | `1` |
| `3` | `pz` | `1` |
| `4` | `x` position (ray-traced onto the unit-cube surface if `proj_ray=True`) | `1` |
| `5` | `y` | `1` |
| `6` | `z` | `1` |
| `7` | raw `pdgid` (mapped to a vocab index inside `LEGOLtng.convert_pdgids`) | `0` |

The FM model only ever sees `model_in = f[..., 1:7]` (the 6-d momentum+position
block) and `pdgids = f[..., 7]` separately. Column 0 is a passthrough slot.

`_F(f)` exposes named views (all are plain tensor slices, not copies):

| View | Slice | Description |
|---|---|---|
| `d` | `f[..., 0, 1]` | density scalar (row 0, col 1) |
| `edep` | `f[..., 1, 1]` | E_dep / E_in scalar (row 1, col 1) |
| `pdgids` | `f[..., -1:]` | pdgid column, all rows |
| `non_p` | `f[..., :2, :]` | both scalar rows, all cols |
| `in_p` | `f[..., 2:3, :]` | incoming-particle row, all cols |
| `out_p` | `f[..., 3:, :]` | outgoing-particle rows, all cols |
| `non_cc` / `in_cc` / `out_cc` | `[..., {rows}, 1:7]` | the 6-d momentum+position block of each row group |
| `model_in` | `f[..., 1:7]` | the 6-d block for every row (what the vector field sees) |
| `mom` | `f[..., 1:4]` | momentum only (3 cols: `px, py, pz`) |

The forward output of `LEGOLtng` / `GenerateOut` is laid out as
`[density_broadcast(1) | solved_model_in(6) | pdgid(1)]` — i.e. the model's
6-d output occupies cols 1–6 with a broadcast density in col 0, mirroring the
on-disk layout. Hence the same `_F` views (which read cols 1–6 of each row,
and row 0 col 1 for `d`) continue to apply to returned samples.

---

## Config reference

The config dict has eight top-level sections (`dl_conf`, `val_conf`,
`base_conf`, `model_conf`, `mm_conf`, `opt_conf`, `odeint_conf`, `additional`).
Any key not listed defaults to the value shown in the source.

### `dl_conf` — dataloader

| Key | Default | Effect |
|---|---|---|
| `lds_args.data` | — | Folder containing `data_prepped.pt` + `meta.json`, or a `.pt` path directly (suffix `.pt` is the discriminator — folder paths get `/data_prepped.pt` appended). |
| `lds_args.cutoff_mev` | `10.0` | Drop outgoing particles with momentum magnitude below this MeV. Also used by `GenerateBase` as the log-floor in the magnitude prior. |
| `lds_args.min_particles` | `0` | Drop events with fewer than this many valid outgoing particles. |
| `lds_args.dtype` | `torch.float32` | Cast features to this dtype. |
| `is_filtered` | `False` | If `True`, the on-disk file is loaded as-is via `get_filtered` (no cutoff applied). |
| `bs` | `2**12` | Batch size. |
| `num_workers` | `4` | `DataLoader` workers (uses `fork` start method if >0). |

### `val_conf` — validation split

Consumed by `LEGOLtng.setup`, which carves a held-out split off the loaded
training set via `random_split`.

| Key | Default | Effect |
|---|---|---|
| `val_frac` | `0.01` | Fraction of the dataset held out for validation (at least 1 event). |
| `seed` | `0` | Seed for the `random_split` generator — reproducible train/val partition. |

### `base_conf` — flow base distribution

| Key | Default | Effect |
|---|---|---|
| `base_dist` | `"poles"` | Only `"poles"` is currently supported (samples vMF around the incoming-particle direction and its antipode). |
| `kappa` | `tensor(10.)` | vMF concentration. Higher → tighter around the pole. |
| `bs_frac` | `0.0` | Fraction of samples placed at the antipodal pole (backscatter). |
| `tanh_theta` | `False` | Use `π·tanh(N(0,1)/κ)` instead of wrapped-normal θ. |
| `scale_dist` | `"trunc_norm"` | Magnitude prior for the momentum scalar: `"trunc_norm"`, `"uniform"`, or `"sm_norm"`. |
| `e_dep_max` | `1.0` | Sigmoid scale for the sampled E_dep base value. |

### `model_conf` — flow-matching model

Top-level FM options:

| Key | Default | Effect |
|---|---|---|
| `manifold` | — | Required. String eval'd in a restricted namespace (only `ProductManifold`, `Euclidean`, `Sphere` are bound), e.g. `"ProductManifold([Euclidean(), Sphere()], (3, 3))"`. The second argument is the per-block ambient dim. |
| `max_energy` | — | Required (here or in `meta.json`). Upper energy bound (MeV) for the `EnergyProjections` normalisation; with `dl_conf.lds_args.cutoff_mev` (lower bound) it maps physical `|p|` to/from the bounded `[0, 1]` energy scalar. `resolve_legoltng_config` raises `KeyError` if it is found in neither `meta.json` nor `model_conf`. |
| `proj_ray` | `True` (read by `DataPrep` only) | At prep-time, ray-trace incoming/outgoing positions onto the unit-cube surface via `CubeTrace`. Ignored when loading already-prepped data from disk. |
| `proj_en` | `False` | Energy normalisation applied in `DataPrep`. Allowed values: `False` / `"identity"` (no-op), `"in_frac"` (divide outgoing momenta by incoming magnitude), `"log"`, `"in_frac_log"`, `"exp"`. (`exp_mult` / `in_mult` exist on `EnergyProjections` but take two arguments and are not callable from this hook.) |
| `ot_coupling` | `False` | At training time, Hungarian-assign base→data per event for OT-style coupling. Requires the optional `torch_lap_cuda_lib` package; otherwise this raises at first call. |
| `ot_e_only` | `False` | When `ot_coupling` is on, base the LAP cost on pairwise `\|p\|` (energy) differences instead of the full 6-D `cdist` — strict energy-ordered pairing. Ignored when `ot_coupling=False`. |
| `cond_cube` | `False` | When solving on the manifold, pass the cube-projected version of the position block as conditioning to the vector field. The integrated state itself stays on the manifold. |
| `t_dist` | `"uniform"` | Time sampling for the loss: `"uniform"`, `"sm_norm"` (`sigmoid(s · N(0,1))`), `"sd3"` (SD3 logit-normal mix `1-u + s/3·((π/2·u).sin()² - u)`), or `"sd3_grid"` (50/50 mix of `"sd3"` with a discrete grid `{0, 0.4, 0.8, 0.9}` — sampled with weights `.1/.2/.3/.4` and jittered by `0.02·N(0,1)`). |
| `t_dist_scale` | `1.4` | Scale `s` for `sm_norm` / `sd3`. |
| `loss_sc` | `0.0` | Weight of an auxiliary "predict-x1" MSE on the momentum 3-vector (`pred = x_t + (1-t)·v`). `0` disables it. |
| `pdgid_is_idx` | `False` | If `True`, the pdgid column is treated as an already-indexed vocab id (skipping `convert_pdgids`). Flipped on by `GenerateOut` at inference. |

`model_conf.model_args` is passed straight to `CFMTrafo_x` and on to the
`x-transformers` Encoder. The wrapper consumes:

| Key | Default | Effect |
|---|---|---|
| `h_dim` | — | Required. Encoder hidden dim. |
| `in_dim` | `6` | Per-particle feature dim — 3 momentum + 3 position. |
| `max_seq_l` | injected from `meta.json` (`ntokens`) | Sequence length used for positional / per-slot embeddings. |
| `nlayers`, `nhead` | `4`, `8` | Encoder depth and heads. |
| `ff_mult` | `1` | Feed-forward expansion factor. |
| `dropout` | `0.1` | Shared attn / ff / emb dropout. |
| `nvtypes` | `2` | Vocab size of the mask-id embedding. The mask tensor only ever contains 0/1, so `2` suffices. |
| `ntypes` | `4` | Vocab size of the per-slot type embedding. The actual indices are `arange(max_seq_l).clamp_max(3)` — so values `0..3` are reached. |
| `npdgids` | injected at training (`len(meta.particles) + 1`) | Vocab size of the pdgid embedding (`+1` for the unknown / pad index `0`). |
| `xavier_gain` | `1.0` | Gain for `xavier_normal_` on the per-embedding linear and bias params. |

All remaining `model_args` keys flow into `x-transformers` `Encoder`, e.g.
`use_adaptive_rmsnorm`, `use_adaptive_layerscale`, `residual_attn`, `ff_swish`,
`ff_glu`, `ff_no_bias`, `gate_residual`, `attn_qk_norm`, `attn_value_rmsnorm`,
`attn_flash`, `rotary_xpos`, …. See `x-transformers` docs for the full list.

### `mm_conf` — multiplicity model

| Key | Default | Effect |
|---|---|---|
| `use_density` | `True` | Concatenate per-event density as an extra input column. If you set this you must bump `in_dim` to match. |
| `in_dim` | `6` | Width of the input projection (`Linear(in_dim, h_dim)`). Must equal the runtime feature dim — 7 if `use_density=True`, otherwise 6. |
| `h_dim` | `512` | Hidden dim of the Decoder. |
| `n_layers`, `n_heads` | `6`, `8` | Decoder depth and heads. |
| `dropout` | `0.1` | Shared attn / ff / layer / emb dropout. |
| `pos_scale` | `50.0` | Multiplier on the position (last-3) part of the input before projection. |
| `max_out_particles` | `meta.ntokens - 3` | Cap on per-pdg-type counts during data loading. |
| `max_count` | derived from data | Categorical vocab size of each per-slot count head; computed once from the train set. |
| `ptypes` | `meta.particles` (sorted) | Outgoing pdg-id vocabulary (`torch.tensor`). One Decoder slot per entry. |
| `ptypes_in` | `meta.particles_in` (sorted) | Incoming pdg-id vocabulary; used for the input embedding. |
| `bs` | `2**12` | Batch size. |
| `lr`, `weight_decay`, `warmup_steps` | `1e-3`, `0.0`, `0` | Used only by the *default* `AdamWScheduleFree` (when `opt_conf` is absent). |
| `post_emb_norm` | `True` | Forwarded to `ContinuousTransformerWrapper`. |
| `use_abs_pos_emb` | `True` | Forwarded to `ContinuousTransformerWrapper`. Note the `use_` prefix. |
| `model_args` | `{}` | Forwarded to `x-transformers` `Decoder` (same flag set as the FM encoder). |
| `opt_conf` | `None` | Same schema as the FM-level `opt_conf` below (resolved by `build_optimizer`). If set, the flat `lr` / `weight_decay` / `warmup_steps` keys above are ignored and a `warnings.warn` is emitted for each. |

### `opt_conf` — optimizer (FM model)

Resolved by `build_optimizer`. Two shapes are accepted:

```python
# (a) Class / callable
"opt_conf": { "opt": schedulefree.AdamWScheduleFree, "lr": 1e-3, "weight_decay": 1e-2 }

# (b) Registry string
"opt_conf": {
    "opt": "muon",               # or "schedulefree"
    "lr": 1e-2, "momentum": 0.95, "nesterov": True, "ns_steps": 5,
    "weight_decay": 1e-2, "weight_decouple": True,
    "adamw_lr": 3e-3, "adamw_betas": (0.9, 0.999), "adamw_wd": 1e-2, "adamw_eps": 1e-8,
    "scheduler": {
        "cls": "warmup_cosine", "total_steps": N, "warmup_frac": 0.05,
        "eta_min": 1e-6, "interval": "step",
    },
}
```

`"muon"` builds a parameter-group `Muon` (2D+ params get Muon; 1D params get
AdamW), `"schedulefree"` builds `AdamWScheduleFree`. `"warmup_cosine"` builds a
`SequentialLR(LinearLR → CosineAnnealingLR)`. Pass a class directly to bypass
the registry.

### `odeint_conf` — inference-time ODE solve

Used by `LEGOLtng.forward` / `LEGOLtng.solve`:

| Key | Default | Effect |
|---|---|---|
| `method` | `"midpoint"` | Any method supported by `flow_matching.solver.ODESolver` (`"midpoint"`, `"rk4"`, `"euler"`, …). |
| `step_size` | `0.04` | ODE step. `0.5` + `"midpoint"` triggers a hand-unrolled 2-step fast path. |
| `split_size` | `None` | Chunk the batch dim when solving to bound memory. |
| `return_timesteps` | `False` | Return intermediate states on a uniform time grid. |
| `return_base` | `False` | Skip the solve and return the sampled base directly. |
| `fwd_compile` | `False` | `torch.compile(model, mode="reduce-overhead")` once on first forward. |
| `filter_pdgid` | `None` | Tensor of PDG-ids to retain; others are NaN-ed. |

`LEGOLtng.solve(..., compute_ll=True, log_p0=...)` runs
`ODESolver.compute_likelihood` (reverse-time) to score samples.

### `additional`

Free-form bag for logging only (`epochs`, `precision`, `notes`,
`comet_exp_key`, …); ignored by the model.

---

## Package layout

| Path | Purpose |
|---|---|
| `legofmt/cfm/cfm_trafo_x.py` | `CFMTrafo_x`: vector field. Embeds `(state, vtype, type-idx, pdgid)` into hidden dim, runs `x-transformers` Encoder conditioned on a sinusoidal `t`-embedding, projects back to `in_dim`. |
| `legofmt/main/modules.py` | `LEGOLtng` Lightning wrapper (loss, sampling base, ODE solve, optional likelihood). `ProjectModel` wraps the vf to project state/velocity onto the manifold. |
| `legofmt/main/generate.py` | `GenerateOut`: chains `MultModel` + `LEGOLtng` for end-to-end sampling, builds the conditioning batch from per-event scalars. |
| `legofmt/main/optimizers.py` | `build_optimizer` registry. Strings `"muon"` and `"schedulefree"` resolve to factories; `"warmup_cosine"` scheduler. Pass classes directly to bypass the registry. |
| `legofmt/multiplicity/model.py` | Autoregressive Decoder over PDG-id slots producing per-type particle counts. |
| `legofmt/data/dataloaders.py` | `GetLEGOData` (energy-cutoff + sort + NaN-handling) and `LEGODataset` (str / dict / tuple constructor, collates to `DataStruct`). |
| `legofmt/data/prep.py` | `DataPrep`: applies `EnergyProjections`, `CubeTrace` ray projection, and `manifold.projx`, then prepends per-event scalar rows. |
| `legofmt/data/struct.py` | `DataStruct(f, m, am)` with views: `f.d` density, `f.edep`, `f.pdgids`, `f.in_p`, `f.out_p`, `f.in_cc`, `f.out_cc`, `f.model_in`, `f.mom`. |
| `legofmt/geometry/path_sample_mult.py` | `ProductManifold` (block-split expmap/logmap/projx/proju) and `ProductPathSampler` (`GeodesicProbPath` per block, `CondOTScheduler`). |
| `legofmt/geometry/gen_base.py` | `GenerateBase`: samples the FM base distribution (vMF "poles" only; magnitude from `scale_dist`). |
| `legofmt/geometry/vmf_sampling.py` | von-Mises–Fisher sampling on the sphere; cartesian↔spherical; `to_cube` projection. |
| `legofmt/geometry/raytracing_proj.py` | `CubeTrace`: ray-trace a position onto the unit cube along its momentum direction. |
| `legofmt/geometry/energy_proj.py` | `EnergyProjections`: energy normalisations — `identity`, `in_frac`, `log`, `in_frac_log`, `exp`, plus the two-argument `in_mult` / `exp_mult` inverses. |
| `legofmt/physics/energy.py`, `legofmt/viz/*` | Physics helpers and corner/rotation plots (not used at train time). |
