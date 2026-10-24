"""Train the per-layer E_dep autoencoder that a flow config's ``model_conf.edep_ae.path`` points at.

It is fitted on that flow's own training split (``LEGOLtng.setup``: same filtering, ``val_conf`` split), so the
flow's validation events stay unseen, and on the events with a deposit. Afterwards the latent standardisation
(``z_mu``, ``z_sd``) and the per-layer draw ceiling (``log_e_max``) are fitted, and ``{"config", "state_dict"}``
is written to the path. Defaults are the 2026-10-20 probe's (K = 16: the smallest K whose sampled Si layers and Si
sum matched G4 to 1-2%; ~1 min on one H100).

    python scripts/train_edep_ae.py <flow config> [--K 16] [--steps 20000]
"""

import argparse
import copy
import os
import time
from pathlib import Path

import yaml

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("config", type=Path, help="flow config (configs/<name>.yaml or a path)")
parser.add_argument("--K", type=int, default=16)
parser.add_argument("--hidden", type=int, default=512)
parser.add_argument("--n-mix", type=int, default=8)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--bs", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent
cfg_path = args.config if args.config.suffix else args.config.with_suffix(".yaml")
if not cfg_path.is_file():
    cfg_path = ROOT / "configs" / cfg_path
config = yaml.safe_load(cfg_path.read_text())["config"]
if (ROOT / ".env").is_file():
    for _l in (ROOT / ".env").read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))

import torch

from legofmt.cfm.edep_ae import LOG_OFFSET, EdepLayerAE
from legofmt.data.prep import DataPrep
from legofmt.main.modules import LEGOLtng

out_path = config["model_conf"]["edep_ae"]["path"]
flow_conf = copy.deepcopy(config)
del flow_conf["model_conf"]["edep_ae"]          # the AE does not exist yet
flow_conf["base_conf"]["kappa"] = torch.tensor(flow_conf["base_conf"]["kappa"])
if "adamw_betas" in flow_conf["opt_conf"]:
    flow_conf["opt_conf"]["adamw_betas"] = tuple(flow_conf["opt_conf"]["adamw_betas"])
if flow_conf["opt_conf"].get("scheduler"):     # the flow's optimiser is built but never stepped here
    flow_conf["opt_conf"]["scheduler"].setdefault("total_steps", 1)
flow_conf["dl_conf"]["lds_args"]["data"] = os.environ.get("LEGO_DATA_DIR", "./data/") + flow_conf["dl_conf"]["lds_args"]["data"]

lego = LEGOLtng(flow_conf)
lego.setup("fit")                               # no trainer attached: the base head is not pretrained here


def _layers(ds):
    """The normalised per-layer E_dep of a (Subset of a) (_Interacting view of a) LEGODataset."""
    if hasattr(ds, "indices"):
        return _layers(ds.dataset)[torch.as_tensor(ds.indices)]
    if hasattr(ds, "keep_idx"):
        return _layers(ds.ds)[ds.keep_idx]
    return ds.data.edep_layers


dev = "cuda"
train = DataPrep(lego.rc.config).denorm_edep(_layers(lego._train_ds).float())
train = train[train.sum(1) > 0].to(dev)
print(f"{train.shape[0]} training events with a deposit, {train.shape[1]} layers", flush=True)

torch.manual_seed(0)
ae = EdepLayerAE(train.shape[1], args.K, hidden=args.hidden, n_mix=args.n_mix).to(dev)
opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
g = torch.Generator(device=dev).manual_seed(1)
t0 = time.time()
for s in range(args.steps):
    loss = ae.nll(train[torch.randint(0, train.shape[0], (args.bs,), device=dev, generator=g)])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
    opt.step()
    sched.step()
    if s % 2500 == 0 or s == args.steps - 1:
        print(f"  step {s:6d} {time.time() - t0:5.0f}s  nll {loss.item():.4f}", flush=True)

ae.eval()
with torch.no_grad():
    sub = train[torch.randperm(train.shape[0], device=dev)[:1_000_000]]
    z = torch.cat([ae.enc(torch.log(e + LOG_OFFSET)) for e in sub.split(65536)])
    ae.z_mu.copy_(z.mean(0))
    ae.z_sd.copy_(z.std(0).clamp(min=1e-6))
    ae.log_e_max.copy_(torch.log(train.max(0).values.clamp(min=1e-12)))

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
torch.save({"config": {"n_layers": train.shape[1], "K": args.K, "hidden": args.hidden, "n_mix": args.n_mix},
            "state_dict": {k: v.cpu() for k, v in ae.state_dict().items()}}, out_path)
print(f"saved {out_path}", flush=True)
