import pytorch_lightning as ltng
import schedulefree
import torch
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.geometry.path_sample_mult import ProductManifold
from legofmt.main.modules import LEGOLtng

d_dtype = torch.float32
torch.set_default_dtype(d_dtype)
torch.set_float32_matmul_precision("medium")

epochs = 30
prec = 32

name = "rp_lq_ar_cls_231125"

flow_man = ProductManifold([Euclidean(), Sphere()], (3, 3))

config = {
    "dl_conf": {
        "path": "./rp_hmgns_lq_ar_varen_e_dep_ft_241025.pt",
        "cutoff_mev": 10,
        "min_particles": 1,
        "include_add": True,
        "is_filtered": True,
        "bs": 2**12,
        "num_workers": 0,
        "dtype": d_dtype,
    },
    "base_conf": {
        "base_range": 3.4,
        "kappa": torch.tensor(40.0),
        "bs_frac": 0.0,
        "base_dist": "poles",
        "scale_dist": "trunc_norm",
    },
    "model_conf": {
        "h_dim": 2**7,
        "ntokens": 12 + 1,
        "in_dim": 6,
        "nlayers": 4,
        "nhead": 8,
        "dropout": 0.1,
        "ff_mult": 2,
        "ff_swish": True,
        "manifold": flow_man,
        "proj_ray": True,
        "ot_coupling": False,
        "proj_en": "in_frac_log",
    },
    "opt_conf": {"opt": schedulefree.AdamWScheduleFree, "lr": 1e-3},
    "additional": {
        "epochs": epochs,
        "precision": str(prec) + ", " + torch.get_float32_matmul_precision(),
    },
}

# lego_flow_model = torch.compile(LEGOLtng(config))
lego_flow_model = LEGOLtng(config)

trainer = ltng.Trainer(
    max_epochs=epochs,
    num_nodes=1,
    accelerator="gpu",
    devices=4,
    precision=prec,
    strategy="ddp",
    check_val_every_n_epoch=1,
)

trainer.fit(model=lego_flow_model)

torch.save(
    {
        "state_dict": lego_flow_model.model.vf.state_dict(),
        "config": config,
    },
    f"./models/cube_flow/{name}.pt",
)
