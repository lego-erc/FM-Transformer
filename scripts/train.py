import argparse
import os
from pathlib import Path

import yaml

parser = argparse.ArgumentParser(description="Train from a YAML config (see configs/).")
parser.add_argument("config", type=Path, help="config name (configs/<name>.yaml) or path to a YAML file")
args = parser.parse_args()

cfg_path = args.config if args.config.suffix else args.config.with_suffix(".yaml")
if not cfg_path.is_file():
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / cfg_path

cfg = yaml.safe_load(cfg_path.read_text())
run, log_conf, config = cfg["run"], cfg["logging"], cfg["config"]

for _l in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    if "=" in _l and not _l.lstrip().startswith("#"):
        _k, _v = _l.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))

if log_conf["comet"]:
    import comet_ml  # noqa: F401  must precede torch/lightning for auto-logging

import lightning as ltng
import torch
from legofmt.main.modules import LEGOLtng
from legofmt.multiplicity.model import MultModel

d_dtype = getattr(torch, run["dtype"])
torch.set_default_dtype(d_dtype)
torch.set_float32_matmul_precision(run["matmul_precision"])

epochs = run["epochs"]
devices = run["devices"]
name = run["name"]

# Coerce the YAML-native values into what the models expect.
config["dl_conf"]["dtype"] = d_dtype
config["base_conf"]["kappa"] = torch.tensor(config["base_conf"]["kappa"])
if "adamw_betas" in config["opt_conf"]:
    config["opt_conf"]["adamw_betas"] = tuple(config["opt_conf"]["adamw_betas"])

dpath_prefix = os.environ.get("LEGO_DATA_DIR", "./data/")
config["dl_conf"]["lds_args"]["data"] = dpath_prefix + config["dl_conf"]["lds_args"]["data"]

scheduler = config["opt_conf"].get("scheduler")
if scheduler is not None and "total_steps" not in scheduler:
    bs = config["dl_conf"]["bs"]
    scheduler["total_steps"] = epochs * int(run["dataset_size"] / (bs * len(devices)))

if log_conf["comet"]:
    from lightning.pytorch.loggers import CometLogger

    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        project=log_conf["project"],
        workspace=os.environ.get("COMET_WORKSPACE"),
        mode="get_or_create",
        name=name,
    )
else:
    logger = False

config["additional"]["epochs"] = epochs
config["additional"]["precision"] = (
    str(run["precision"]) + ", " + torch.get_float32_matmul_precision()
)
config["additional"]["comet_exp_key"] = logger._experiment_key if logger else None

if logger:
    logger.log_hyperparams(config)

trainer = ltng.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    devices=devices,
    precision=run["precision"],
    strategy=run["strategy"],
    logger=logger,
    val_check_interval=run["val_check_interval"],
    limit_val_batches=run["limit_val_batches"],
    gradient_clip_val=run["gradient_clip_val"],
)

train_model = run["train_model"]
compile_mode = run["compile"]  # false | model
if train_model == "fm":
    model = LEGOLtng(config)
    if compile_mode == "model":
        model.model = torch.compile(model.model, dynamic=False)
else:
    model = MultModel(config)

trainer.fit(
    model=model
)

model.rc.config["dl_conf"]["lds_args"]["data"] = "<dataset_path>"
model.rc.config["dl_conf"]["data_path"] = None
model.rc.config["additional"]["comet_exp_key"] = None

if train_model == "fm":
    vf = model.model._orig_mod.vf if compile_mode == "model" else model.model.vf
    state_dict, ckpt_dir = vf.state_dict(), "./checkpoints/flow/"
else:
    state_dict, ckpt_dir = model.state_dict(), "./checkpoints/mult/"

torch.save(
    {"state_dict": state_dict, "config": model.rc.config},
    f"{os.environ.get('LEGO_CKPT_DIR', ckpt_dir)}{name}.pt",
)
