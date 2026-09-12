"""GPU-free preflight for a training config. Catches everything that fails before
the first CUDA kernel -- data paths, device/allocation mismatch, ckpt_dir, config
resolution, model construction -- in ~30s, so you don't queue for a GPU just to
discover a typo.

    CONDA_OVERRIDE_CUDA=13.0 pixi run -e train python scripts/preflight.py <config>

The override is only needed where no GPU is visible (login node): the cuda-gated
environment declares cuda = "13.0", so a lower value still fails.
Exit code 0 = clear to submit; 1 = at least one blocking problem.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FAIL, WARN = [], []


def bad(m):  FAIL.append(m); print(f"  FAIL  {m}")
def warn(m): WARN.append(m); print(f"  warn  {m}")
def ok(m):   print(f"  ok    {m}")


ap = argparse.ArgumentParser()
ap.add_argument("config", help="bare name (-> configs/<name>.yaml) or a path")
ap.add_argument("--build", action="store_true",
                help="also construct the model on CPU (slower; catches shape errors)")
args = ap.parse_args()

cfg_path = Path(args.config)
cfg_path = cfg_path if cfg_path.suffix else cfg_path.with_suffix(".yaml")
if not cfg_path.is_file():
    cfg_path = ROOT / "configs" / cfg_path
if not cfg_path.is_file():
    print(f"  FAIL  config not found: {args.config}"); sys.exit(1)
ok(f"config {cfg_path}")

env_file = ROOT / ".env"
if env_file.is_file():
    for _l in env_file.read_text().splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    ok(f".env read from {env_file}")
else:
    warn(f"no .env at {env_file}; LEGO_DATA_DIR/LEGO_CKPT_DIR fall back to defaults")

cfg = yaml.safe_load(cfg_path.read_text())
run, config = cfg["run"], cfg["config"]
train_model = run.get("train_model", "fm")
print(f"\n[run] name={run.get('name')}  train_model={train_model}  epochs={run.get('epochs')}")

# ---- data -----------------------------------------------------------------
prefix = os.environ.get("LEGO_DATA_DIR", "./data/")
dpath = Path(prefix + config["dl_conf"]["lds_args"]["data"])
print(f"\n[data] LEGO_DATA_DIR={prefix}")
meta = None
if not dpath.is_dir():
    bad(f"data dir missing: {dpath}")
else:
    ok(f"data dir {dpath}")
    mp, pp = dpath / "meta.json", dpath / "data_prepped.pt"
    if not mp.is_file():
        bad(f"meta.json missing: {mp}")
    else:
        meta = json.loads(mp.read_text())
        ok(f"meta.json: ntokens={meta['ntokens']} species={len(meta['particles'])} "
           f"particles_in={meta.get('particles_in')} max_energy={meta.get('max_energy')}")
    if not pp.is_file():
        bad(f"data_prepped.pt missing: {pp}")
    else:
        gb = pp.stat().st_size / 1e9
        ok(f"data_prepped.pt {gb:.1f} GB")
        nranks = len(run.get("devices", [0])) or 1
        need = gb * nranks
        warn(f"LEGODataset torch.loads this per rank with no mmap: "
             f"~{need:.0f} GB across {nranks} rank(s) -- size --mem accordingly")

mc = config.get("model_conf", {})
if meta and train_model == "fm" and "max_energy" in mc:
    if float(mc["max_energy"]) != float(meta.get("max_energy", -1)):
        bad(f"max_energy {mc['max_energy']} != dataset's {meta.get('max_energy')}")
    else:
        ok(f"max_energy matches dataset ({mc['max_energy']})")

# ---- devices vs allocation ------------------------------------------------
dev = run.get("devices")
print(f"\n[devices] config devices={dev}")
if isinstance(dev, list):
    ntpn = os.environ.get("SLURM_NTASKS_PER_NODE")
    if ntpn and int(ntpn) != len(dev):
        bad(f"len(devices)={len(dev)} != SLURM_NTASKS_PER_NODE={ntpn} (DDP will hang or idle GPUs)")
    elif ntpn:
        ok(f"len(devices)={len(dev)} matches SLURM_NTASKS_PER_NODE")
    else:
        warn(f"len(devices)={len(dev)}; not under SLURM here -- it must equal "
             f"--ntasks-per-node AND --gres=gpu:N on the target node")
    if max(dev) >= len(dev):
        warn(f"device indices go up to {max(dev)}; a node with only {len(dev)} GPUs "
             f"exposes 0..{len(dev)-1} and Lightning will refuse")
    if run.get("strategy") == "ddp" and len(dev) == 1:
        warn("strategy: ddp with a single device pulls in NCCL for nothing; use 'auto'")

# ---- checkpoint destination ----------------------------------------------
ckpt_dir = run.get("ckpt_dir") or os.path.join(
    os.environ.get("LEGO_CKPT_DIR", "./checkpoints/"), "flow" if train_model == "fm" else "mult")
print(f"\n[ckpt] dir={ckpt_dir}")
try:
    os.makedirs(ckpt_dir, exist_ok=True)
    probe = Path(ckpt_dir) / ".preflight_write_probe"
    probe.touch(); probe.unlink()
    ok("ckpt_dir exists and is writable")
except Exception as e:
    bad(f"ckpt_dir not writable ({type(e).__name__}: {e}) -- this fails AFTER training completes")
target = Path(ckpt_dir) / f"{run.get('name')}.pt"
if target.exists():
    warn(f"{target.name} already exists and WILL be overwritten (train.py does a bare torch.save)")

# ---- schedule / walltime -------------------------------------------------
sched = config.get("opt_conf", {}).get("scheduler")
if sched is not None and "total_steps" not in sched and isinstance(dev, list):
    bs = config["dl_conf"]["bs"]
    ts = run["epochs"] * int(run["dataset_size"] / (bs * len(dev)))
    ok(f"total_steps={ts} (epochs {run['epochs']} x dataset {run['dataset_size']} / (bs {bs} x {len(dev)}))")

# ---- optional-dependency gates ------------------------------------------
print("\n[deps]")
if train_model == "fm" and mc.get("ot_coupling"):
    try:
        import torch_lap_cuda_lib  # noqa: F401
        ok("ot_coupling=True and torch_lap_cuda_lib importable")
    except ImportError:
        bad("ot_coupling=True but torch_lap_cuda_lib missing -> on_fit_start raises. "
            "Run `pixi run -e <env> install-lap` (needs __cuda: use a GPU node)")
if cfg.get("logging", {}).get("comet"):
    try:
        import comet_ml  # noqa: F401
        ok("comet: true and comet_ml importable")
    except ImportError:
        bad("comet: true but comet_ml missing -- use a *-comet environment")
if train_model == "mult" and "max_count" not in config.get("mm_conf", {}):
    warn("mm_conf.max_count unset: config resolution builds a MultLoader, i.e. an "
         "extra full load of data_prepped.pt before training starts")

# ---- resolve (+ optionally build) on CPU --------------------------------
if not FAIL:
    print("\n[resolve]")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        config["dl_conf"]["lds_args"]["data"] = str(dpath)
        if sched is not None and "total_steps" not in sched and isinstance(dev, list):
            sched["total_steps"] = run["epochs"] * int(
                run["dataset_size"] / (config["dl_conf"]["bs"] * len(dev)))
        if train_model == "fm":
            from legofmt.mod_comps.config import resolve_legoltng_config
            rc = resolve_legoltng_config(config)
            ok(f"resolved: max_seq_l={rc.max_seq_l} npdgids={rc.model_args['npdgids']} "
               f"n_prefix={rc.n_prefix} cond_scalars={rc.cond_scalars}")
            if args.build:
                from legofmt.main.modules import LEGOLtng
                m = LEGOLtng(config)
                ok(f"model built on CPU: {sum(p.numel() for p in m.parameters()):,} params")
        else:
            from legofmt.mod_comps.config import resolve_mult_config
            rc = resolve_mult_config(cfg if "state_dict" in cfg else {"config": config})
            ok(f"resolved: species={rc.max_seq_len} max_particles={rc.max_particles} "
               f"in_dim={rc.in_dim} ptypes_in={rc.ptypes_in.tolist()}")
            if args.build:
                from legofmt.multiplicity.model import MultModel
                m = MultModel({"config": config})
                ok(f"model built on CPU: {sum(p.numel() for p in m.parameters()):,} params")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")

print(f"\n==== {len(FAIL)} blocking, {len(WARN)} warning(s) ====")
sys.exit(1 if FAIL else 0)
