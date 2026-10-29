# legofmt

Riemannian continuous flow-matching transformer that generates the outgoing particles of a calorimeter-style
(LEGO/Geant4) shower event from the incoming particle and the material (density, atomic and mass number, size).
Each event has one incoming particle and a variable number of outgoing particles labelled by PDG id.

Generation is **two-stage**:

1. **Multiplicity model** (`MultModel`, an autoregressive `x-transformers` Decoder) draws how many outgoing particles
   of each PDG id the event has, and optionally a pass-through token ("the primary did not interact").
2. **Flow-matching model** (`LEGOLtng` wrapping the vector field `CFMTrafo_x`, an `x-transformers` Encoder)
   integrates an ODE on `Euclidean(1) × Sphere(3) × Sphere(3)` (energy scalar, momentum direction, surface-position
   direction) for that many outgoing slots, plus the event's `E_dep`.

Both are `lightning.LightningModule`s configured by one nested `config` dict and saved as `{"state_dict", "config"}`.

---

## Quickstart — generate an event

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
size    = torch.tensor([100.0])              # material size (the Size scalar) (B,) or (1,)
Z, A    = torch.tensor([29.0]), torch.tensor([63.5])  # atomic / mass number  (B,) or (1,)
pdgids  = torch.tensor([11])                 # a PDG-id, e^- in this case;    (B,) or (1,)

with torch.no_grad():
    out = gen.gen_model_w_g4_args(n, pos, mom, energy, density, size, pdgids, Z=Z, A=A)

# out["per_particle"]["Outgoing"] : [B*n, max_seq_l - n_prefix - 1, 8]   model layout (see Data structure)
# out["per_particle"]["Incoming"] : [B*n, 1, 8]
# out["per_event"]                : {"E_dep", *cond_scalars}   -- E_dep in model normalisation, not MeV
# out["per_voxel"]                : {"E_dep": empty [B*n, 0, 4]}
# out["per_slab"]                 : {"E_dep": per-layer E_dep}, only for a flow with model_conf.edep_ae
print(out["per_particle"]["Outgoing"].shape)
```

Each input has batch size `1` or `B`; `n` samples are drawn per event, so the output has `B*n` rows. `mom` is a
**direction**: it is normalised and scaled by `energy` internally, and `pos` is ray-traced onto the conditioning
cube (the raw `gen(cond)` path below needs energy-scaled momentum). Every name in the flow's `cond_scalars` must be
supplied -- `Density`, `Z`, `A`, `Size` for current checkpoints; a legacy `("Density",)` checkpoint ignores `Z` and
`A`, and one without `Size` takes a single `size`. `passthrough=False` skips the pass-through token;
`passthrough_prob_w_g4_args(...)` returns its probability.

---

## Workflow

```
raw event dict ─► GetLEGOData ─► DataPrep.prep ─► folder/{data_prepped.pt, meta.json}        scripts/gen_data.py
                  (cutoff, NaN,    (projx, ray trace,
                   sort)            scalar rows)

folder ─► LEGODataset(folder, prep=DataPrep(cfg)) ─► LEGOLtng / MultModel ─► {"state_dict", "config"}   scripts/train.py
          (DataPrep.norm_e scales the MeV channels at load)

flow + multiplicity checkpoints ─► GenerateOut: sample counts, solve the ODE on the product manifold
```

`LEGODataset` dispatches on `data`: a folder or `.pt` path loads the prepped tuple (normalised by `prep.norm_e`
when a `prep` is given); a raw dict needs `prep=DataPrep(config)` and runs `GetLEGOData` + `DataPrep.prep`.

### Training

`python scripts/train.py <config>` (`configs/fm_default.yaml`, `configs/mult_default.yaml`) is the entry point.
Devices, precision, compile, seeds (`init_seed`, `data_seed`), logging (Comet, or a CSV logger via
`logging.csv_dir`), `ckpt_dir` and resuming (`resume_from`: weights; `resume_ckpt`: a Lightning checkpoint with
optimizer state) come from the YAML's `run` / `logging` sections. `warmup_cosine`'s `total_steps` is the trainer's
exact step count (`run.dataset_size` only sizes a construction placeholder); an optional `run.max_steps` ends
training after that many optimizer steps if `run.epochs` has not, and the schedule then spans exactly those steps.
`scripts/preflight.py <config> [--build]` checks data, devices, `ckpt_dir` and config resolution on CPU before you
queue for a GPU; `scripts/submit_slrm.sh <config>` submits a run to SLURM.

The same thing by hand:

```python
import lightning as ltng
import torch
from legofmt.cfm.project_model import uncompiled
from legofmt.main.modules import LEGOLtng

model = LEGOLtng(config)                      # see the config reference below
ltng.Trainer(max_epochs=10, accelerator="gpu", devices=[0, 1, 2, 3], strategy="ddp").fit(model)
model._opt_eval()                             # schedule-free optimizers: switch to the averaged weights
vf_sd = uncompiled(model.model).vf.state_dict()   # model.model is a ProjectModel around the CFMTrafo_x field
torch.save({"state_dict": vf_sd, "config": model.rc.config}, "flow.pt")
```

`scripts/train.py` also writes the frozen base head (`base_conf.base_head`) and the learned loss weights
(`model_conf.loss_weights`) into the saved config; a `MultModel` saves its whole `state_dict()`. `LEGOLtng(ckpt)` /
`MultModel(ckpt)` reload from the loaded dict (a bare config builds a fresh model).

### Generation — lower-level API

Underneath `gen_model_w_g4_args` (see [Quickstart](#quickstart--generate-an-event)) sits the raw `cond` API:

```python
from legofmt.main.generate import GenerateOut

gen = GenerateOut("flow.pt", "mult.pt", device="cuda",
                  couple_in_out_pdgids=False)  # True: NaN out generated slots whose pdgid is not one of
                                               # the multiplicity model's incoming species (filter_pdgid)

# Raw call: cond is [B, n_cond + 7] = [*cond_scalars, px,py,pz (energy-scaled), x,y,z, pdgid_raw]
sols, mask, attn_mask = gen(cond)              # gen(cond, prepped=False, passthrough=True)
# sols : [B, ntokens, 8]  in the model layout: [energy scalar, mom dir(3), pos dir(3), pdgid_raw]
# mask / attn_mask : [B, ntokens]
```

After the solve `GenerateOut` clamps E_dep and applies the optional `odeint_conf.energy_balance` /
`exit_consistency`. `GenerateIn` runs the reverse direction (incoming PID from the outgoing set) and needs a
`train_inverse` multiplicity checkpoint.

---

## Data structure

### On disk

Each prepared dataset is a folder containing:

- `data_prepped.pt` — the tuple `(f, m, am)` of `DataStruct`, plus a fourth tensor `edep_layers` (per-layer E_dep)
  when the raw data has one. The two `max_energy`-dependent channels (incoming energy scalar, `E_dep`) are stored in
  MeV and normalised at load time by `DataPrep.norm_e`.
- `meta.json` — `ntokens`, `particles` (sorted outgoing PDG-ids), `particles_in` (incoming PDG-ids the multiplicity
  model knows), `max_energy`, `cutoff_mev`, `cond_scalars`, `energy_kin`, `cuboid_dim`, `n_edep_layers` and
  `edep_excess_max_mev` (per incoming PDG-id, the largest `E_dep - E_in`: the `edep_wall` margin).

`ntokens = n_prefix + 1 + max_outgoing`, where `n_prefix = len(cond_scalars) + 1`
(the conditioning-scalar rows plus the generated `E_dep` row).

### In memory (`DataStruct`)

`DataStruct(f, m, am, edep_layers=None)` wraps the tensors. `N = ntokens`.

| Field | Shape | Meaning |
|---|---|---|
| `f` (features) | `[B, N, 8]` | Per-slot features, see layout below. |
| `m` (loss mask) | `[B, N]` int | `1` where the slot is a random variable the flow must produce; `0` where it is a condition or pad. |
| `am` (attn mask) | `[B, N]` bool | `True` for valid slots (transformer attention mask). |

Row layout along `N` (slot roles depend on `model_conf.cond_scalars`, default
`("Density",)`; current configs use `("Density", "Z", "A", "Size")`):

```
row 0              : cond_scalars[0] (Density)                    mask=0  attn=1
row 1              : E_dep (normalised) -- generated, not a condition    mask=1  attn=1
rows 2..n_prefix-1 : cond_scalars[1:] (Z, A, Size, ...)           mask=0  attn=1
row n_prefix       : incoming particle (condition)                mask=0  attn=1
rows n_prefix+1..  : outgoing particles (RVs, padded)             mask=1  attn=1/0
```

Column layout along the last dim of `f` (8 columns):

| Col | Particle rows | Scalar rows |
|---|---|---|
| `0` | energy scalar: `log(E/cutoff)/log(max_energy/cutoff)` clamped to `[0, 1]`; **outgoing rows store `1 - log(E_out/cutoff)/log(E_in/cutoff)`** (relative to the incoming particle; `E_out >= E_in` clamps to `0`) | the scalar's value |
| `1:4` | unit momentum direction | `1` |
| `4:7` | unit position direction (ray-traced onto the cube surface for the incoming row if `proj_ray=True`) | `1` |
| `7` | pdgid (raw, or a vocab index when `pdgid_is_idx`) | `0` |

`model_in = f[..., 0:7]` — seven columns, and `model_args.in_dim` must be 7. Col 0 is
the flow's energy channel (`Euclidean(1)` factor), not a passthrough.

Always index through `_F(f)` / `DataStruct` (plain slices, not copies); the
layout is process-global state set by `set_layout(cond_scalars)`:

| View | Slice | Description |
|---|---|---|
| `d` / `edep` | `f[..., 0, 0]` / `f[..., 1, 0]` | density and `E_dep` scalars |
| `cond(name)` | `f[..., cond_slot(name), 0]` | any conditioning scalar by name |
| `pdgids` | `f[..., -1:]` | pdgid column, all rows |
| `non_p` / `in_p` / `out_p` | rows `[:n_prefix]` / `[n_prefix]` / `[n_prefix+1:]`, all cols | scalar rows, incoming, outgoing |
| `non_cc` / `in_cc` / `out_cc` | same rows, cols `0:7` | the 7-d model block of each row group |
| `model_in` / `energy` | `f[..., 0:7]` / `f[..., 0:1]` | what the vector field sees / its energy channel |

`LEGOLtng.forward` / `GenerateOut` return the same `[B, N, 8]` layout (solved
`model_in` in cols `0:7`, pdgid in col 7, conditioning rows passed through, NaN
outside `attn_mask`), so the same `_F` views apply to generated samples.

---

## Config reference

The config dict has eight top-level sections (`dl_conf`, `val_conf`,
`base_conf`, `model_conf`, `mm_conf`, `opt_conf`, `odeint_conf`, `additional`).
Any key not listed defaults to the value shown in the source.

### `dl_conf` — dataloader

| Key | Default | Effect |
|---|---|---|
| `lds_args.data` | — | Folder with `data_prepped.pt` + `meta.json` (fresh training requires the folder; `LEGODataset` also loads a bare `.pt` path). |
| `lds_args.cutoff_mev` | from `meta.json` | Lower bound of the energy normalisation (`log(E/cutoff)/log(max_energy/cutoff)`); the file's value wins. The cut itself is applied at prep time (kinetic energy when `energy_kin`, else `\|p\|`, incoming row included). |
| `lds_args.min_particles` | `0` | Prep time (raw-dict path): drop events with fewer valid outgoing particles. |
| `lds_args.frac` | `1.0` | Keep a seeded random fraction of the events. |
| `lds_args.dtype` | `torch.float32` | Prep time (raw-dict path): cast features to this dtype. |
| `bs` | `2**12` | Batch size. |
| `num_workers` | `4` | `DataLoader` workers (uses `fork` start method if >0). |
| `seed` | `run.data_seed`, else Lightning's global seed, else `0` | Shuffle seed, identical on every DDP rank (`loader_seed`). |
| `len_buckets`, `len_mode`, `len_split`, `len_max_tokens` | `None`, `"split"`, `8`, `None` | Length buckets (`"auto"` = powers of two): run each batch (`"group"`: one bucket per batch, `len_max_tokens` events x length per batch) or each sorted piece of it (`"split"`, `len_split` pieces) at its bucket's padded length. |

### `val_conf` — validation split

`LEGOLtng.setup` / `MultModel.setup` carve a held-out split off the loaded training set via `random_split`.

| Key | Default | Effect |
|---|---|---|
| `val_frac` | `0.01` | Fraction of the dataset held out for validation (at least 1 event). |
| `seed` | `0` | Seed for the `random_split` generator — reproducible train/val partition. |
| `count_floor` | `0.005` | `MultModel` only: species with a lower mean count are left out of `val/count_mae`. |

### `base_conf` — flow base distribution

| Key | Default | Effect |
|---|---|---|
| `base_dist` | `"poles"` | Direction prior: `"poles"` (vMF-like around the incoming direction), `"iso"` (isotropic momentum and position), `"iso_pos"` (poles for momentum, isotropic position). |
| `kappa` | `tensor(10.)` | Concentration. Higher → tighter around the pole. Overwritten per event by the learned `base_head` when present. |
| `bs_frac` | `0.0` | `"poles"` only: fraction of events whose outgoing base positions (not momenta) sit around the antipode of the incoming position (backscatter). |
| `tanh_theta` | `False` | Use `π·tanh(N(0,1)/κ)` instead of wrapped-normal θ. |
| `scale_dist` | `"trunc_norm"` | Energy prior: `"trunc_norm"`, `"uniform"`, `"sm_norm"` (`1 - tanh(|N(0,1)|·sm_scale)`), `"logit_norm"`. The learned `base_head` requires `"sm_norm"`. |
| `sm_scale` | `0.5` | `sm_norm` tanh temperature; larger → flatter energy base. Per event from `base_head` when present. |
| `e_dep_max` | `1.0` | Sigmoid scale for the sampled E_dep base value. |
| `base_head` | — | Saved `base_head` state dict (written by training). Reloaded frozen — the head is never trained alongside the flow — unless its shape no longer matches the config (`base_head_nout`, `base_head_hidden`, `ang_*` toggled), in which case a fresh head is pretrained. |
| `base_head_hidden` | `[16]` | Hidden widths of the head (16 under-fits E_dep's prior). |
| `base_head_nout` | `False` | Also feed the outgoing multiplicity and per-pdgid composition to the head; only E_dep's `(mu, sig)` see them. |
| `ang_shape`, `ang_nout`, `ang_species`, `ang_weight` | `False`, `False`, `False`, `1.0` | Angular prior (needs `tanh_theta`): a shape exponent next to `kappa` (two matched moments); `kappa` sees the outgoing set; one `(kappa, pow)` per outgoing species, fitted per slot; weight of the angular moment loss. |

### `model_conf` — flow-matching model

Top-level FM options:

| Key | Default | Effect |
|---|---|---|
| `manifold` | — | Required. List of factor dicts whose dims sum to 7, e.g. `[{name: euclidean, dim: 1}, {name: sphere, dim: 3}, {name: sphere, dim: 3}]`. The old eval'd string form (`"ProductManifold([...], (3, 3))"`) still loads with a `DeprecationWarning`. |
| `max_energy` | — | Required (here or in `meta.json`). Upper energy bound (MeV) for the `EnergyProjections` normalisation; with `dl_conf.lds_args.cutoff_mev` (lower bound) it maps physical energy to/from the bounded `[0, 1]` energy scalar. A pure normalisation ceiling: it may exceed the gun energy (E_dep can exceed T_kin). |
| `cond_scalars` | from `meta.json`, else `("Density",)` | Per-event conditioning scalars; sets the row layout (see Data structure). |
| `energy_kin` | from `meta.json`, else `True` | Use Geant4's recorded kinetic energy as the energy scalar instead of `|p|` (which saturates for hadrons). |
| `edep_log_min` | `None` | Log-scale the E_dep row: `log(E_dep/edep_log_min)/log(max_energy/edep_log_min)`, clamped to `[0, 1]`; `None` keeps `E_dep/max_energy`. |
| `overflow_delta` | `0.0` | Sentinel offset: zero-valued outgoing-energy cells (the full-energy atom, `E_out >= E_in`) and zero-deposit E_dep targets are written to `-overflow_delta`, so the flow can separate the atom from the continuum. |
| `proj_ray` | `True` (read by `DataPrep` only) | At prep-time, ray-trace the incoming position onto the unit-cube surface via `CubeTrace`. |
| `canon_sym` | `False` | Canonicalise directions onto a reference cube face before the flow and undo it after (`CubeSymmetry`). Cube only (`cuboid_dim` with unequal sides raises). |
| `sym_aug` | `False` | With `canon_sym`: a random in-face symmetry per event on top of the canonicalisation (training and generation). |
| `cuboid_dim` | from `meta.json`, else a cube | Target dimensions: the ray trace and the position direction (that of `x / cuboid_dim`). |
| `exclude_passthrough` | `False` | Train only on events where the primary interacted (`E_dep > 0` or `n_out != 1`); the count model's pass-through token emits the rest. |
| `edep_cell`, `max_loss_weight_edep` | `False`, `= max_loss_weight` | Give the E_dep row its own loss cell; under `learned_loss_weights` its own learned weight and floor. |
| `edep_ae` | `None` | `{path: <ae.pt>}` from `scripts/train_edep_ae.py <flow config>`: per-layer E_dep (`E_dep_layers` in the data) through the K latents of a frozen autoencoder (`legofmt/cfm/edep_ae.py`), flown with the particles as one extra token of the field (`model_args.lat_dim = K`, own loss cell `loss/lat`). `GenerateOut` decodes them after energy balancing into `edep_layers` (MeV, summing to the E_dep row) and `per_slab` of the G4-shaped output; the flow checkpoint carries the AE's weights. Not with training `len_buckets` in `"split"` mode, reflow, inverse-mask events or `log_likelihood`. |
| `edep_wall`, `edep_wall_species` | `False`, `None` (all) | E_dep row as the log-odds of the deposited fraction of `E_cap = E_in + margin(species)`: `log E - log(E_cap - E)` mapped from `[log(edep_log_min / E_cap), log(E_cap / 1e-3 MeV)]` to `[0, 1]`. Small deposits keep a log scale, a contained shower's peak just below `E_in` gets a wide channel (the 1 GeV e- peak spans 0.0024 of the plain log row), and `E_cap` is the edge, so `GenerateOut` skips its per-species E_dep clamp. The margin is `model_conf.edep_excess_mev`, set from `meta.json`'s `edep_excess_max_mev` (`data/prep.py:EDEP_WALL_MARGIN_MEV` for older data). `edep_wall_species` restricts the wall to those incoming pdgids; the rest keep the log row. The per-layer tensor keeps the plain scale. Train with it. |
| `edep_whiten`, `edep_whiten_max_sig`, `edep_whiten_feed`, `edep_whiten_global` | `False`, `None`, `False`, `None` | E_dep row in units of the frozen base head's per-event prior, `y = (logit(u / e_dep_max) - mu) / sig` (needs the base head; not with `log_likelihood`); a cap on that unit so a broad prior keeps a fine MeV grid; hand each event's `(mu, unit)` to the network; one global `(c, s)` map for every event instead. |
| `dir_zoom` | `None` | `theta_s` (rad). Learn each outgoing momentum direction with its polar angle about the incoming direction remapped to `pi*log1p(theta/theta_s)/log1p(pi/theta_s)` (azimuth kept; applied to the targets in training, undone in `forward`), so a particle that crosses a thin plate nearly undeflected (mrad) is resolvable. `2e-3` magnifies 7 mrad ~47x and compresses 1 rad ~2.3x. Train with it. |
| `dir_zoom_mix` | `0.0` | Linear share of the `dir_zoom` map (`0` = pure log). |
| `dir_zoom_nucleons_max` | `None` | With `dir_zoom`: leave events with more outgoing nucleons (p, n) than this unzoomed. `1` skips nuclear interactions (in G4, >= 2 nucleons marks 99.4-99.9% of them in 1.4 mm W, and <= 1 holds for 99.9% of survivor and EM events), whose secondaries spread over the sphere; zooming them cost the W plate ~20% of its nuclear-interaction deposit. Read from the pdgids, which the count model fixes before the flow, so training and generation agree. |
| `ot_coupling` | `False` | At training time, Hungarian-assign base→data slots per event (same pdgid only). Solved with `torch-linear-assignment` (a pixi dependency); `on_fit_start` raises if it is missing. |
| `t_dist` | `"sd3"` | Training-time `t` sampling: `"sd3"` (mode sampling `1-u + s/3·(sin²(πu/2) - u)`; `s = 0` is uniform) or `"sm_norm"` (`sigmoid(s·N(0,1))`, i.e. logit-normal). |
| `t_dist_scale`, `t_dist_shift` | `1.4`, `1.0` | Scale `s` above; `shift != 1` applies `t ** (1/shift)`. |
| `mask_conf.p_forward` | `0.5` if `mask_conf` is set | Per-event coin flip between the forward mask and its inverse (generate the incoming from the outgoing set); no `mask_conf` = all forward. |
| `learned_loss_weights`, `max_loss_weight`, `max_loss_weight_flow` | `False`, `-6.0`, `= max_loss_weight` | Kendall-style learned per-cell log-variance weights `exp(-lv)·L + lv` (energy / dir / pos, plus the E_dep and latent cells when on), clamped at the floor(s), i.e. `weight_bound = e^-max_loss_weight` (403x at `-6`); the flow-map term has its own cell and floor. Pre-2026-09 `uncert_*` keys are migrated on load. |
| `one_step_euler_fac`, `one_step_euler_sections`, `one_step_euler_every` | `0.0`, `8`, `1` | Flow-map self-distillation weight, dyadic ladder depth, and step gating. `> 0` requires `model_args.step_cond`. |
| `base_pretrain_batches`, `base_pretrain_bs` | `300`, `dl_conf.bs` | Learned base head: up-front pretraining batches, after which the head is frozen — it is never trained alongside the flow. Needs `Z`, `A`, `Size` in `cond_scalars` and `scale_dist: sm_norm`. |
| `reflow_path`, `reflow_start_epoch`, `reflow_every`, `reflow_kwargs` | `None`, `0`, `1`, `{}` | Reflow against a frozen teacher every `reflow_every` steps: a `reflow_path` teacher from step 0; without one, `reflow_start_epoch > 0` turns on self-reflow (the student's snapshot, refreshed each epoch from that epoch on). |
| `pdgid_is_idx` | `False` | If `True`, the pdgid column is treated as an already-indexed vocab id (skipping `convert_pdgids`). Flipped on by `GenerateOut` at inference. |

`model_conf.model_args` is passed straight to `CFMTrafo_x` and on to the
`x-transformers` Encoder. The wrapper consumes:

| Key | Default | Effect |
|---|---|---|
| `h_dim` | `256` | Encoder hidden dim. |
| `in_dim` | `6` (configs set `7`) | Per-slot feature dim: energy scalar + 3 momentum + 3 position = 7; must equal the manifold's total dim. |
| `max_seq_l` | `meta.json` `ntokens` unless set | Sequence length used for the per-slot type indices. |
| `nlayers`, `nhead` | `4`, `8` | Encoder depth and heads. |
| `ff_mult` | `1` | Feed-forward expansion factor. |
| `dropout` | `0.1` | Shared attn / ff / emb dropout. |
| `nvtypes` | `2` | Vocab size of the mask-id conditional map. The mask tensor only ever contains 0/1, so `2` suffices. |
| `ntypes` | `len(cond_scalars) + 3` unless set | Vocab size of the per-slot type map; indices are `arange(max_seq_l).clamp_max(n_prefix + 1)`. |
| `npdgids` | set at fresh resolve: `len(meta.particles) + 1` | Vocab size of the pdgid conditional map (`+1` for the unknown / pad index `0`). |
| `xavier_gain` | `1.0` | Gain on the init std of the conditional linear maps. |
| `time_cond` | `True` | Sinusoidal `t` embedding as the adaptive-norm condition (conditioning tokens at `t = 1`); `False` uses a learned global vector (`LEGOLtngDirect`). With one `t` per event the adaptive norms compute their gammas on two rows per event instead of one per token. |
| `step_cond` | `False` | Also embed the step size `d` (zero-init gain), required by `one_step_euler_fac > 0`; needs `time_cond`. |
| `grad_ckpt` | `False` | Activation checkpointing per attention block. |
| `geom_encoder` | `None` | Layered-stack conditioning: `{layers: [[thickness mm, density, Z, A], ...] (beam side first), repeat: 1, tokens: 4, dim: 128, depth: 3, heads: 4, hide_scalars: true}`. A small bidirectional transformer over one token per layer (plus X0 / lambda_I thicknesses and Fourier features of the cumulative depth) prepends `tokens` summary tokens in `h_dim` to the sequence (conditioning tokens, dropped before the output projection; zero-init read-out). `hide_scalars` masks the composite Density / Z / A / Size rows out of the flow's attention (they stay in the data for the base head and the count model). One stack per model for now (`legofmt/cfm/geom_encoder.py`). |

All remaining `model_args` keys flow into `x-transformers` `Encoder`, e.g.
`use_adaptive_rmsnorm`, `use_adaptive_layerscale`, `residual_attn`, `ff_swish`,
`ff_glu`, `ff_no_bias`, `gate_residual`, `attn_qk_norm`, `attn_value_rmsnorm`,
`attn_flash`, `rotary_xpos`, …. See `x-transformers` docs for the full list.

`attn_qk_norm_scale` is version-dependent: x-transformers < 2.25.5 applied it
three times (logits `scale³·cosθ`, i.e. `1000·cosθ` at the default 10), newer
versions once. Every config records `additional.x_transformers_version`; a
checkpoint without it (or with an older one) gets its scale cubed at load so it
reproduces exactly, while fresh configs use the library default of 10.

### `mm_conf` — multiplicity model

| Key | Default | Effect |
|---|---|---|
| `cond_scalars` | from `meta.json`, else `("Density",)` | Per-event conditioning scalars prepended to the incoming token; the input width is derived as `len(cond_scalars) + 7`. |
| `max_energy`, `edep_log_min` | from `meta.json`, `None` | Normalisation of the incoming energy scalar (`GenerateOut` rebases the input when the flow's ceiling differs) and of E_dep, as for the flow. |
| `canon_sym` | `False` | Canonicalise the incoming direction onto a reference cube face (`proj_in`). |
| `train_inverse` | `False` | Also train `InvModel` (incoming PID from the outgoing set), needed by `GenerateIn`; `inv_h_dim`, `inv_n_layers`, `inv_n_heads`, `inv_model_args` default to the count model's. |
| `h_dim` | `512` | Hidden dim of the Decoder. |
| `n_layers`, `n_heads` | `6`, `8` | Decoder depth and heads. |
| `dropout` | `0.1` | Shared attn / ff / emb dropout (layer dropout is not plumbed: it breaks DDP). |
| `pos_scale` | `50.0` | Half-size `d` of the L-infinity cube `GeomTrafos.to_cube` maps the position direction onto before the input projection. |
| `sym_aug` | `False` | With `canon_sym`: random in-face symmetry per event. |
| `passthrough_head`, `passthrough_pdgids`, `passthrough_on`, `passthrough_dim`, `passthrough_depth` | `False`, the neutrals, `"in_embd"`, `h_dim`, `0` | Pass-through token: P(the primary did not interact) for the listed incoming species, read from the input embedding or the trunk's slot 0, an MLP (`depth 0`) or `depth` residual blocks; the count CE is masked for those events. |
| `max_out_particles` | `meta.ntokens - (n_prefix + 1)` | Counts per pdg type are clamped below this during data loading. |
| `max_count` | derived from data | Categorical vocab size of each per-slot count head: the largest count in the dataset + 1, computed once at fresh resolve (`max_out_particles` for checkpoints without it). |
| `ptypes` | `meta.particles` (sorted) | Outgoing pdg-id vocabulary (`torch.tensor`). One Decoder slot per entry. |
| `ptypes_in` | `meta.particles_in` (sorted) | Incoming pdg-id vocabulary; used for the input embedding. |
| `bs` | `2**12` | The loader's batch size; must equal `dl_conf.bs` (a fresh config raises otherwise). |
| `lr`, `weight_decay`, `warmup_steps` | `1e-3`, `0.0`, `0` | Used only by the *default* `AdamWScheduleFree` (when there is no `opt_conf`). |
| `post_emb_norm` | `True` | Forwarded to `ContinuousTransformerWrapper`. |
| `use_abs_pos_emb` | `True` | Forwarded to `ContinuousTransformerWrapper`. Note the `use_` prefix. |
| `model_args` | `{}` | Forwarded to `x-transformers` `Decoder` (same flag set as the FM encoder). |
| `opt_conf` | `None` | Read only when the top-level `opt_conf` is absent (back-compat); same schema. Either one replaces the flat `lr` / `weight_decay` / `warmup_steps` keys above. |
| `fwd_compile` | `False` | `GenerateOut` `torch.compile(dynamic=True)`s the count Decoder at load. Dynamic because the KV-cache decode changes the cache length every slot: a static graph recompiled per step (68 vs 115 ms per 8192 events). Not `reduce-overhead`: CUDA-graph capture rejects the reused KV cache. Counts identical under a fixed seed. |

### `opt_conf` — optimizer (both models)

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

`"muon"` builds `BatchedMuon` (2D+ params get Muon; 1D params get AdamW),
`"schedulefree"` builds `AdamWScheduleFree`. `"warmup_cosine"` builds a
`SequentialLR(LinearLR → CosineAnnealingLR)`; under a Trainer `configure_optimizers`
sets its `total_steps` to `trainer.estimated_stepping_batches`. Pass a class
directly to bypass the registry.

### `odeint_conf` — inference-time ODE solve

Used by `LEGOLtng.forward` / `LEGOLtng.solve`:

| Key | Default | Effect |
|---|---|---|
| `method` | `"midpoint"` | `"euler"`, `"midpoint"`, `"rk4"` are hand-unrolled and manifold-aware (only `mask == 1` slots move; euler and midpoint's first stage get the flow map's `d`); any other `flow_matching.solver.ODESolver` method runs in the ambient space. |
| `step_size` | `0.04` | ODE step: `round(1/step_size)` equal steps. |
| `time_grid` | `None` | Explicit grid (list or tensor), used as given. |
| `amp` | `additional.precision` | Autocast dtype of the solve (`"bf16"`, `"32"`, ...). |
| `len_buckets`, `compile_buckets` | `None` | Solve events grouped by length at their bucket's padded length (a list, or `"auto"`: powers of two; `max_seq_l` is always one); pad chunk row counts to `compile_buckets` sizes so a compiled field sees few shapes. |
| `sym_rand` | `False` | With `canon_sym`: random in-face symmetry at generation even without `sym_aug`. |
| `split_size` | `None` | Chunk the batch dim when solving to bound memory. |
| `return_timesteps` | `False` | Return the state at every time-grid point (initial state included). |
| `return_base` | `False` | Skip the solve and return the sampled base directly. |
| `fwd_compile` | `False` | `torch.compile(model, mode="reduce-overhead")` once on first forward (no CUDA graphs under `compile_dynamic` without `compile_buckets`). |
| `compile_dynamic` | `False` | With `fwd_compile`: one symbolic-shape graph instead of one per (length bucket, row count). Patches x-transformers' shape-specialising `einx.where` (`compat.dynamic_shape_attention`), hands the field contiguous inputs and tiles 1-row chunks (dynamo would recompile on either). Slab chain: 2 graphs for the W/Si pair instead of 28, ~1 min of compile instead of ~5. |
| `filter_pdgid` | `None` | Tensor of PDG-ids to retain: other outgoing slots are NaN-ed and their pdgid set to 0 (pads are kept). Set by `GenerateOut(couple_in_out_pdgids=True)`. |
| `energy_balance` | `False` | `GenerateOut` only. Keep the generated E_dep and close the kinetic balance `E_in = E_dep + sum E_out + 1.022 (n_e+ out - n_e+ in)` per event: e+-/gamma in rescale every outgoing energy (an absorbed one deposits the budget), a hadron survivor (the leading particle is the incoming species with > 90% of E_in, or only gammas / e+- come out beside it) takes the remainder, nuclear interactions are left alone. Only for data whose own balance closes (0.1 MeV-cutoff plates; a 10 MeV cutoff loses ~7% below it). Set per model via lego_eval's `odeint` override. |
| `exit_consistency` | `False` | `GenerateOut` only. Mirror the momentum component of an outgoing particle that points into the face its exit position is on. Position and direction are separate channels, so the flow can emit one leaving +z while moving -z (Geant4 never does); a slab chain hands it to the next slab moving away from it, which the count model answers with spurious species. |

`LEGOLtng.log_likelihood(ds_t, log_p0, step_size, method)` runs
`ODESolver.compute_likelihood` (reverse-time) to score samples; step size and method are its arguments, not
`odeint_conf`, and it raises under `edep_ae` or `edep_whiten`. `solve` and `log_likelihood` work in the model frame:
a `canon_sym` / `dir_zoom` caller must transform the batch first, as `forward` does.

### `additional`

Mostly a logging bag (`epochs`, `notes`, `comet_exp_key`, `git_rev`, …). Read back at load: `precision` (the
default `odeint_conf.amp`) and `x_transformers_version` (the qk-norm scale above).

---

## Package layout

| Path | Purpose |
|---|---|
| `legofmt/cfm/cfm_trafo_x.py` | `CFMTrafo_x`: vector field. Per-(mask, type, pdgid) conditional linear embedding, `x-transformers` Encoder conditioned on sinusoidal `t` (and `d`) embeddings, mirrored output projection. |
| `legofmt/cfm/project_model.py`, `legofmt/cfm/solvers.py`, `legofmt/cfm/path_sampler.py` | `ProjectModel` keeps state/velocity on the manifold; `Solvers` mixin (`solve`, hand-unrolled euler / midpoint / rk4 steps, `log_likelihood`, `forward`); `ProductPathSampler` (the CondOT geodesic per factor, closed form). |
| `legofmt/cfm/edep_ae.py`, `legofmt/cfm/geom_encoder.py` | `EdepLayerAE`, the per-layer E_dep autoencoder (`model_conf.edep_ae`); `GeomEncoder`, the layered-stack summary tokens (`model_args.geom_encoder`). |
| `legofmt/main/modules.py` | `LEGOLtng`: construction, Lightning hooks, data split, optimizer wiring. Behaviour lives in the mixins `TrainStep` / `BaseDist` / `Solvers`. |
| `legofmt/main/train_step.py` | `TrainStep`: training step, per-cell uncertainty weighting, `t` and mask sampling, flow-map loss gating. |
| `legofmt/main/generate.py` | `GenerateOut` / `GenerateIn`: chain `MultModel` + `LEGOLtng` for end-to-end sampling; build the padded conditioning batch. |
| `legofmt/base_dist/gen_base.py`, `legofmt/base_dist/base_nn.py` | `GenerateBase` samplers (`poles` / `iso` / `iso_pos`, `scale_dist`) and the learned per-event `base_head` (`BaseDist` mixin, pretraining, OT coupling). |
| `legofmt/distill/distill.py`, `legofmt/distill/reflow.py` | Flow-map self-distillation loss; reflow teacher and the one-step `LEGOLtngDirect` / `GenerateOutDirect` variants. |
| `legofmt/mod_comps/config.py`, `legofmt/mod_comps/optimizers.py` | Config resolution (`Resolved*Config`, manifold building, version stamps) and the optimizer registry (`BatchedMuon`, `"schedulefree"`, `"warmup_cosine"`). |
| `legofmt/multiplicity/model.py` | `MultModel`: autoregressive Decoder over PDG-id slots producing per-type particle counts, with the pass-through head; optional `InvModel`. |
| `legofmt/data/dataloaders.py` | `LEGODataset` (energy cutoff, sorting, NaN handling via `GetLEGOData`; collates to `DataStruct`), the length-bucket batch sampler and `make_loader`. |
| `legofmt/data/prep.py` | `DataPrep`: energy normalisation, `CubeTrace` ray projection, `manifold.projx`, the per-event scalar rows, `norm_e`/`norm_edep`. |
| `legofmt/data/struct.py` | `DataStruct(f, m, am)`, the `_F` views, and the process-global `set_layout`. |
| `legofmt/geometry/*` | `ProductManifold`, `GeomTrafos` (direction sampling, cartesian↔spherical, `to_cube`), `CubeTrace`, `EnergyProjections`, `CubeSymmetry`. |
| `legofmt/log_metrics/val_metrics.py` | `ShowerValMetrics`: MMD and per-feature W1 on particle kinematics and event summaries (vendored from `lego-eval`). |
| `legofmt/compat.py` | Loading older checkpoints: parameter renames, qk-norm scale, fp32 attention, legacy manifold strings; the dynamic-shape attention patch. |
| `legofmt/viz/*` | Corner plots and cube plots (not used at train time). |
| `scripts/` | `train.py`, `preflight.py`, `submit_slrm.sh`, `gen_data.py` (Geant4 via `pyg4lego`), `train_edep_ae.py`. |
