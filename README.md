# legofmt

Riemannian continuous flow-matching transformer for generating outgoing particles
given an incoming particle and a material density. Targets calorimeter-style
shower data (LEGO/Geant4) where each event has one incoming particle and a
variable number of outgoing particles labelled by PDG-id.

Generation is **two-stage**:

1. **Multiplicity model** (`MultModel`, autoregressive `x-transformers` decoder)
   predicts how many outgoing particles of each PDG-id the event contains.
2. **Flow-matching model** (`LEGOLtng` wrapping `CFMTrafo_x`, an `x-transformers`
   encoder) integrates an ODE on `Euclidean(1) × Sphere(3) × Sphere(3)` (energy
   scalar, momentum direction, surface-position direction) to produce the
   per-particle kinematics for that many outgoing slots, plus the event's `E_dep`.

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

# out["per_particle"]["Outgoing"] : [B*n, max_seq_l - n_prefix - 1, 8]   model layout (see Data structure)
# out["per_particle"]["Incoming"] : [B*n, 1, 8]
# out["per_event"]                : {"E_dep", *cond_scalars}   -- E_dep in model normalisation, not MeV
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

# Raw call: cond is [B, n_cond + 7] = [*cond_scalars, px,py,pz (energy-scaled), x,y,z, pdgid_raw]
sols, mask, attn_mask = gen(cond)
# sols : [B, ntokens, 8]  in the model layout: [energy scalar, mom dir(3), pos dir(3), pdgid_raw]
# mask / attn_mask : [B, ntokens]
```

---

## Data structure

### On disk

Each prepared dataset is a folder containing:

- `data_prepped.pt` — a 3-tuple `(target, mask, attn_mask)` matching `DataStruct.__init__`.
  The two `max_energy`-dependent channels (incoming energy scalar, `E_dep`) are
  stored in MeV and normalised at load time by `DataPrep.norm_e`.
- `meta.json` — `ntokens`, `particles` (sorted outgoing PDG-ids), `particles_in`
  (incoming PDG-ids the multiplicity model knows), `max_energy`, `cutoff_mev`,
  `cond_scalars`, `energy_kin`.

`ntokens = n_prefix + 1 + max_outgoing`, where `n_prefix = len(cond_scalars) + 1`
(the conditioning-scalar rows plus the generated `E_dep` row).

### In memory (`DataStruct`)

`DataStruct(f, m, am)` wraps three tensors. `N = ntokens`.

| Field | Shape | Meaning |
|---|---|---|
| `f` (features) | `[B, N, 8]` | Per-slot features, see layout below. |
| `m` (loss mask) | `[B, N]` int | `1` where the slot is a random variable the flow must produce; `0` where it is a condition or pad. |
| `am` (attn mask) | `[B, N]` bool | `True` for valid slots (transformer attention mask). |

Row layout along `N` (slot roles depend on `model_conf.cond_scalars`, default
`("Density",)`; current configs use `("Density", "Z", "A", "Size")`):

```
row 0              : cond_scalars[0] (Density)                    mask=0  attn=1
row 1              : E_dep / max_energy  -- generated, not a condition   mask=1  attn=1
rows 2..n_prefix-1 : cond_scalars[1:] (Z, A, Size, ...)           mask=0  attn=1
row n_prefix       : incoming particle (condition)                mask=0  attn=1
rows n_prefix+1..  : outgoing particles (RVs, padded)             mask=1  attn=1/0
```

Column layout along the last dim of `f` (8 columns):

| Col | Particle rows | Scalar rows |
|---|---|---|
| `0` | energy scalar: `log(E/cutoff)/log(max_energy/cutoff)` clamped to `[0, 1]`; **outgoing rows store `1 - e_out/e_in`** (relative to the incoming particle) | the scalar's value |
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
| `lds_args.data` | — | Folder containing `data_prepped.pt` + `meta.json`, or a `.pt` path directly (suffix `.pt` is the discriminator — folder paths get `/data_prepped.pt` appended). |
| `lds_args.cutoff_mev` | from `meta.json` | Lower bound of the energy normalisation (`log(E/cutoff)/log(max_energy/cutoff)`). Prepped datasets are already cut; the cutoff itself is applied at prep time (kinetic energy when `energy_kin`, else `\|p\|`, incoming row included). |
| `lds_args.min_particles` | `0` | Drop events with fewer than this many valid outgoing particles (prep time). |
| `lds_args.frac` | `1.0` | Keep a seeded random fraction of the events. |
| `lds_args.dtype` | `torch.float32` | Cast features to this dtype. |
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
| `base_dist` | `"poles"` | Direction prior: `"poles"` (vMF-like around the incoming direction, `bs_frac` at the antipode), `"iso"` (isotropic momentum and position), `"iso_pos"` (poles for momentum, isotropic position). |
| `kappa` | `tensor(10.)` | Concentration. Higher → tighter around the pole. Overwritten per event by the learned `base_head` when present. |
| `bs_frac` | `0.0` | Fraction of samples placed at the antipodal pole (backscatter). |
| `tanh_theta` | `False` | Use `π·tanh(N(0,1)/κ)` instead of wrapped-normal θ. |
| `scale_dist` | `"trunc_norm"` | Energy prior: `"trunc_norm"`, `"uniform"`, `"sm_norm"` (`1 - tanh(|N(0,1)|·sm_scale)`), `"logit_norm"`. The learned `base_head` requires `"sm_norm"`. |
| `sm_scale` | `0.5` | `sm_norm` tanh temperature; larger → flatter energy base. Per event from `base_head` when present. |
| `e_dep_max` | `1.0` | Sigmoid scale for the sampled E_dep base value. |
| `base_head` | — | Saved `base_head` state dict (written by training); always reloaded frozen — the head is never trained alongside the flow. |
| `base_head_nout` | `False` | Also feed the outgoing multiplicity and per-pdgid composition to the head; only E_dep's `(mu, sig)` see them. |

### `model_conf` — flow-matching model

Top-level FM options:

| Key | Default | Effect |
|---|---|---|
| `manifold` | — | Required. List of factor dicts whose dims sum to 7, e.g. `[{name: euclidean, dim: 1}, {name: sphere, dim: 3}, {name: sphere, dim: 3}]`. The old eval'd string form (`"ProductManifold([...], (3, 3))"`) still loads with a `DeprecationWarning`. |
| `max_energy` | — | Required (here or in `meta.json`). Upper energy bound (MeV) for the `EnergyProjections` normalisation; with `dl_conf.lds_args.cutoff_mev` (lower bound) it maps physical energy to/from the bounded `[0, 1]` energy scalar. A pure normalisation ceiling: it may exceed the gun energy (E_dep can exceed T_kin). |
| `cond_scalars` | from `meta.json`, else `("Density",)` | Per-event conditioning scalars; sets the row layout (see Data structure). |
| `energy_kin` | from `meta.json`, else `True` | Use Geant4's recorded kinetic energy as the energy scalar instead of `|p|` (which saturates for hadrons). |
| `edep_log_min` | `None` | Log-scale the E_dep row: `log(E_dep/edep_log_min)/log(max_energy/edep_log_min)`, clamped to `[0, 1]`; `None` keeps `E_dep/max_energy`. |
| `overflow_delta` | `0.0` | Sentinel offset: zero-deposit / fully-absorbed targets are written to `-overflow_delta` in the energy channel so the flow can separate the atom from the continuum. |
| `proj_ray` | `True` (read by `DataPrep` only) | At prep-time, ray-trace the incoming position onto the unit-cube surface via `CubeTrace`. |
| `canon_sym` | `False` | Canonicalise directions onto a reference cube face before the flow and undo it after (`CubeSymmetry`). |
| `ot_coupling` | `False` | At training time, Hungarian-assign base→data slots per event (same pdgid only). Requires `torch_lap_cuda_lib`; `on_fit_start` raises otherwise. |
| `t_dist` | `"sd3"` | Training-time `t` sampling: `"sd3"` (mode sampling `1-u + s/3·(sin²(πu/2) - u)`; `s = 0` is uniform) or `"sm_norm"` (`sigmoid(s·N(0,1))`, i.e. logit-normal). |
| `t_dist_scale`, `t_dist_shift` | `1.4`, `1.0` | Scale `s` above; `shift != 1` applies `t ** (1/shift)`. |
| `mask_conf.p_forward` | — | If `mask_conf` is set, per-event coin flip between the forward mask and its inverse (generate the incoming from the outgoing set). |
| `learned_loss_weights`, `max_loss_weight`, `max_loss_weight_flow` | `False`, `-6.0`, `= max_loss_weight` | Kendall-style learned per-channel log-variance weights `exp(-lv)·L + lv` (energy / dir / pos), clamped at the floor(s), i.e. `weight_bound = e^-max_loss_weight` (403x at `-6`); the flow-map term has its own cell and floor. Pre-2026-09 `uncert_*` keys are migrated on load. |
| `one_step_euler_fac`, `one_step_euler_sections`, `one_step_euler_every` | `0.0`, `8`, `1` | Flow-map self-distillation weight, dyadic ladder depth, and step gating. `> 0` requires `model_args.step_cond`. |
| `base_pretrain_batches`, `base_pretrain_bs` | `300`, `dl_conf.bs` | Learned base head: up-front pretraining batches, after which the head is frozen — it is never trained alongside the flow. Needs `Z`, `A`, `Size` in `cond_scalars` and `scale_dist: sm_norm`. |
| `reflow_path`, `reflow_start_epoch`, `reflow_every`, `reflow_kwargs` | `None`, `0`, `1`, `{}` | Reflow against a frozen teacher (or the student's own snapshot when no path). **Gated solely by `reflow_start_epoch > 0`.** |
| `pdgid_is_idx` | `False` | If `True`, the pdgid column is treated as an already-indexed vocab id (skipping `convert_pdgids`). Flipped on by `GenerateOut` at inference. |

`model_conf.model_args` is passed straight to `CFMTrafo_x` and on to the
`x-transformers` Encoder. The wrapper consumes:

| Key | Default | Effect |
|---|---|---|
| `h_dim` | — | Required. Encoder hidden dim. |
| `in_dim` | `6` (configs set `7`) | Per-slot feature dim: energy scalar + 3 momentum + 3 position = 7; must equal the manifold's total dim. |
| `max_seq_l` | injected from `meta.json` (`ntokens`) | Sequence length used for the per-slot type indices. |
| `nlayers`, `nhead` | `4`, `8` | Encoder depth and heads. |
| `ff_mult` | `1` | Feed-forward expansion factor. |
| `dropout` | `0.1` | Shared attn / ff / emb dropout. |
| `nvtypes` | `2` | Vocab size of the mask-id conditional map. The mask tensor only ever contains 0/1, so `2` suffices. |
| `ntypes` | injected: `len(cond_scalars) + 3` | Vocab size of the per-slot type map; indices are `arange(max_seq_l).clamp_max(n_prefix + 1)`. |
| `npdgids` | injected at training (`len(meta.particles) + 1`) | Vocab size of the pdgid conditional map (`+1` for the unknown / pad index `0`). |
| `xavier_gain` | `1.0` | Gain on the init std of the conditional linear maps. |
| `time_cond` | `True` | Sinusoidal per-token `t` embedding as the adaptive-norm condition; `False` uses a learned global vector (`LEGOLtngDirect`). |
| `step_cond` | `False` | Also embed the step size `d` (zero-init gain), required by `one_step_euler_fac > 0`. |
| `grad_ckpt` | `False` | Activation checkpointing per attention block. |

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
| `max_energy` | from `meta.json` | Normalisation ceiling for the incoming energy scalar (`GenerateOut` rebases the input when the flow's ceiling differs). |
| `canon_sym` | `False` | Canonicalise the incoming direction onto a reference cube face (`proj_in`). |
| `train_inverse` | `False` | Also train `InvModel` (incoming PID from the outgoing set), needed by `GenerateIn`; `inv_h_dim`, `inv_n_layers`, `inv_n_heads`, `inv_model_args` default to the count model's. |
| `h_dim` | `512` | Hidden dim of the Decoder. |
| `n_layers`, `n_heads` | `6`, `8` | Decoder depth and heads. |
| `dropout` | `0.1` | Shared attn / ff / layer / emb dropout. |
| `pos_scale` | `50.0` | Multiplier on the position (last-3) part of the input before projection. |
| `max_out_particles` | `meta.ntokens - (n_prefix + 1)` | Cap on per-pdg-type counts during data loading. |
| `max_count` | derived from data | Categorical vocab size of each per-slot count head; computed once from the train set. |
| `ptypes` | `meta.particles` (sorted) | Outgoing pdg-id vocabulary (`torch.tensor`). One Decoder slot per entry. |
| `ptypes_in` | `meta.particles_in` (sorted) | Incoming pdg-id vocabulary; used for the input embedding. |
| `bs` | `2**12` | Batch size. |
| `lr`, `weight_decay`, `warmup_steps` | `1e-3`, `0.0`, `0` | Used only by the *default* `AdamWScheduleFree` (when `opt_conf` is absent). |
| `post_emb_norm` | `True` | Forwarded to `ContinuousTransformerWrapper`. |
| `use_abs_pos_emb` | `True` | Forwarded to `ContinuousTransformerWrapper`. Note the `use_` prefix. |
| `model_args` | `{}` | Forwarded to `x-transformers` `Decoder` (same flag set as the FM encoder). |
| `opt_conf` | `None` | Same schema as the FM-level `opt_conf` below (resolved by `build_optimizer`). If set, the flat `lr` / `weight_decay` / `warmup_steps` keys above are ignored and a `warnings.warn` is emitted for each. |
| `fwd_compile` | `False` | Re-apply `torch.compile` to the count Decoder when a saved checkpoint is loaded by `GenerateOut`. `train.py` compiles it during training but unwraps it before saving, so generation otherwise runs eager. Plain compile, not `mode="reduce-overhead"` — the AR loop reuses its KV cache across steps and CUDA-graph capture rejects that. Measured 1.28x on the multiplicity phase at bs 8192; counts bit-identical under a fixed seed. |

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
| `legofmt/cfm/cfm_trafo_x.py` | `CFMTrafo_x`: vector field. Per-(mask, type, pdgid) conditional linear embedding, `x-transformers` Encoder conditioned on sinusoidal `t` (and `d`) embeddings, mirrored output projection. |
| `legofmt/cfm/project_model.py`, `legofmt/cfm/solvers.py`, `legofmt/cfm/path_sampler.py` | `ProjectModel` keeps state/velocity on the manifold; `Solvers` mixin (`solve`, hand-unrolled midpoint/Euler steps, `log_likelihood`, `forward`); `ProductPathSampler` (one `GeodesicProbPath` per factor). |
| `legofmt/main/modules.py` | `LEGOLtng`: construction, Lightning hooks, data split, optimizer wiring. Behaviour lives in the mixins `TrainStep` / `BaseDist` / `Solvers`. |
| `legofmt/main/train_step.py` | `TrainStep`: training step, per-cell uncertainty weighting, `t` and mask sampling, flow-map loss gating. |
| `legofmt/main/generate.py` | `GenerateOut` / `GenerateIn`: chain `MultModel` + `LEGOLtng` for end-to-end sampling; build the padded conditioning batch. |
| `legofmt/base_dist/gen_base.py`, `legofmt/base_dist/base_nn.py` | `GenerateBase` samplers (`poles` / `iso` / `iso_pos`, `scale_dist`) and the learned per-event `base_head` (`BaseDist` mixin, pretraining, OT coupling). |
| `legofmt/distill/distill.py`, `legofmt/distill/reflow.py` | Flow-map self-distillation loss; reflow teacher and the one-step `LEGOLtngDirect` variant. |
| `legofmt/mod_comps/config.py`, `legofmt/mod_comps/optimizers.py` | Config resolution (`Resolved*Config`, manifold building, version stamps) and the optimizer registry (`BatchedMuon`, `"schedulefree"`, `"warmup_cosine"`). |
| `legofmt/multiplicity/model.py` | Autoregressive Decoder over PDG-id slots producing per-type particle counts; optional `InvModel`. |
| `legofmt/data/dataloaders.py` | `LEGODataset` (energy cutoff, sorting, NaN handling; collates to `DataStruct`) and `make_loader`. |
| `legofmt/data/prep.py` | `DataPrep`: energy normalisation, `CubeTrace` ray projection, `manifold.projx`, the per-event scalar rows, `norm_e`/`norm_edep`. |
| `legofmt/data/struct.py` | `DataStruct(f, m, am)`, the `_F` views, and the process-global `set_layout`. |
| `legofmt/geometry/*` | `ProductManifold`, `GeomTrafos` (direction sampling, cartesian↔spherical, `to_cube`), `CubeTrace`, `EnergyProjections`, `CubeSymmetry`. |
| `legofmt/log_metrics/val_metrics.py` | `ShowerValMetrics`: MMD and per-feature W1 on particle kinematics and event summaries (vendored from `lego-eval`). |
| `legofmt/compat.py` | Loading older checkpoints: parameter renames, qk-norm scale, fp32 attention, legacy manifold strings. |
| `legofmt/viz/*` | Corner plots and cube plots (not used at train time). |
