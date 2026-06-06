import comet_ml
import os
from pathlib import Path

for _l in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    if "=" in _l and not _l.lstrip().startswith("#"):
        _k, _v = _l.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))

import lightning as ltng
from lightning.pytorch.loggers import CometLogger
import schedulefree
import torch
from legofmt.main.modules_direct import LEGOLtng
from legofmt.multiplicity.model import MultModel

from pytorch_optimizer import AdEMAMix 
from pytorch_optimizer import Muon                                                                                                         
import torch.optim.lr_scheduler as lrs         
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR                                                                                                                                                                                                                                 

d_dtype = torch.float32
torch.set_default_dtype(d_dtype)
torch.set_float32_matmul_precision("medium")                                                                                                    

ds_scale = 1
epochs = 10 * ds_scale
prec = 32
bs = 2**12
devices = [0, 1, 2, 3]
dataset_size = int(2e7 / ds_scale)

name = "rp_mult_v1_020626"

comet_logger = CometLogger(
    api_key=os.environ["COMET_API_KEY"],
    project="lego_pdgid",
    workspace=os.environ.get("COMET_WORKSPACE"),
    mode="get_or_create",
    name=name,
)

dpath_prefix = os.environ.get("LEGO_DATA_DIR", "./data/")
total_steps = epochs * int(dataset_size / (bs * len(devices)))   

config = {
    "dl_conf": {
        "lds_args": {
            "data": f"{dpath_prefix}rp_lqar_20M_080526",
            "frac": 1. / ds_scale,
            "cutoff_mev": 10,
            "min_particles": 0,
        },
        "is_filtered": False,
        "bs": bs,
        "num_workers": 16,
        "dtype": d_dtype,
    },
    "val_conf": {"val_frac": 0.01, "seed": 0},
    "base_conf": {
        "kappa": torch.tensor(8.),
        "bs_frac": 0.,
        "base_dist": "poles",
        "scale_dist": "sm_norm",
        "tanh_theta": True,
    },
    "model_conf": {
        "manifold": "ProductManifold([Euclidean(), Sphere()], (3, 3))",
        "proj_ray": False, # True,
        "ot_coupling": True,
        "ot_e_only": False,
        "proj_en": "in_frac_log",
        "t_dist": "sd3_grid",
        "t_dist_scale": 0.,
        "loss_sc": 0.,
        "cond_cube": False,
        "model_args": {
            "h_dim": 2**8,
            "in_dim": 6,
            "nlayers": 6,
            "nhead": 8,
            "dropout": 0.02,
            "ff_mult": 4,
            "nvtypes": 2,
            "use_adaptive_rmsnorm": True,
            "use_adaptive_layerscale": True,
            "ff_swish": True,
            "ff_glu": True,
            "ff_no_bias": False,
            "attn_flash": True,
            "attn_qk_norm": True,
            "attn_value_rmsnorm": True,
        },
    },
    "mm_conf": {
        "use_density": True,
        "dropout": 0.1,
        "h_dim": 128,
        "n_layers": 6,
        "n_heads": 6,
        "ff_mult": 4,
        "in_dim": 7,
        "bs": 2**12,
        "lr": 1e-3,
        "label_smoothing": 0.,
        "val_frac": 0.,
        "weight_decay": 1e-2,
        "warmup_steps": 0,
        "ce_focal_gamma": 0.,
        "post_emb_norm": False,
        "use_abs_pos_emb": False,
        "model_args": {
            "ff_swish": True,
            "ff_glu": True,
            "attn_qk_norm": True,
            "rotary_xpos": True,
            "use_adaptive_rmsnorm": True,
            "use_adaptive_layerscale": True,
            "residual_attn": True,
        },
    },
      "opt_conf": {                                                                                                                              
      "opt": "muon",                                                                                                                        
      "lr": 7e-3,                                                                                           
      "momentum": 0.95,                                                                                                                      
      "nesterov": True,                                                                                                                      
      "ns_steps": 5,                                                                                                                         
      "weight_decay": 1e-2,                                              
      "weight_decouple": True,                                                                                                                
      "adamw_lr": 3e-3,                                                                                                                      
      "adamw_betas": (0.9, 0.999),                                                                                                            
      "adamw_wd": 1e-2,                                                                                                                      
      "adamw_eps": 1e-8,                                                                     
    "scheduler": {
        "cls": "warmup_cosine",                                                                                                                  
        "total_steps": total_steps,                                                                                                            
        "warmup_frac": 0.1,
        "eta_min": 1e-6,                                                                                                                       
        "interval": "step",                                                                                                                    
    },                                                                                                                                   
  },      
    # "opt_conf": {
    #     "opt": schedulefree.AdamWScheduleFree,
    #     "lr": 1e-3,
    #     "weight_decay": 1e-2,
    #     "betas": (0.95, 0.999),
    # },                                                                                                                                                                                                                                                                                                                            
    "additional": {
        "epochs": epochs,
        "precision": str(prec) + ", " + torch.get_float32_matmul_precision(),
        "notes": "",
        "comet_exp_key": comet_logger._experiment_key,
    },
}
comet_logger.log_hyperparams(config)

trainer = ltng.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    devices=devices,
    precision=prec,
    strategy="ddp",
    logger=comet_logger,
    val_check_interval=0.25,
    limit_val_batches=4,
    gradient_clip_val=1.,
)

model = LEGOLtng(config)
model.model = torch.compile(model.model, dynamic=False)

trainer.fit(
    model=model
)

# model.opt.eval()

model.rc.config["dl_conf"]["lds_args"]["data"] = "<dataset_path>"
model.rc.config["dl_conf"]["data_path"] = None
model.rc.config["additional"]["comet_exp_key"] = None

torch.save(
    {
        "state_dict": model.model._orig_mod.vf.state_dict(),
        "config": model.rc.config,
    },
    f"{os.environ.get('LEGO_CKPT_DIR', './checkpoints/flow/')}{name}.pt",
)
