# Structural Overview

- ### cfm: Central Continuous Flow Matching Components. 

    - cfm_trafo_x.py: xTransformer construction and Input processing. 

- ### main: Model wrapper and Generate Function.

    - modules.py: Lightning wrapper for training and inference.  
    - generate.py: Generation wrapper running multiplicity model and FM model successively.

- ### data

    - dataloaders.py: gets data from a file and preprocesses (e.g. cutoff energy, filters nan's)

- ### geometry: Various utilities for geometry transformations.

    - energy_proj.py: normalization for the magnitute of the momentum. 
    - gen_base.py: flow base sample generation. 
    - path_sample_mult.py: enables projections and path computations on products of manifolds. 
    - raytracing_proj.py: projects to the other side of the cube, can add noise. 
    - vmf_sampling.py: sampling uniformly on a sphere, transformations from cartesian to spherical.
    
- ### multiplicity

    - model.py: model to generate a distribution for the number of outgoing particles.

- ### Config Dict (exemplary):

    ```
    config = {
        "dl_conf": {
            "lds_args": {
                "path": FOLDERPATH,
                "cutoff_mev": 10,
                "min_particles": 0,
            },
            "is_filtered": True,
            "bs": 2**12,
            "num_workers": 16,
            "dtype": torch.float32,
        },
        "base_conf": {
            "base_range": 3.4,
            "kappa": torch.tensor(4.),
            "bs_frac": 0.,
            "base_dist": "poles",
            "scale_dist": "trunc_norm"
        },
        "model_conf": {
            "manifold": "ProductManifold([Euclidean(), Sphere()], (3, 3))",
            "proj_ray": True,
            "ot_coupling": True,
            "proj_en": "in_frac_log",
            "t_dist": "sm_norm",
            "loss_sc": 1e-5,
            "model_args": {
                "h_dim": 2**9,
                "in_dim": 6,
                "nlayers": 6,
                "nhead": 8,
                "dropout": 0.1,
                "ff_mult": 6,
                "use_adaptive_rmsnorm": True,
                "use_adaptive_layerscale": True,
                "residual_attn": True,
                "ff_swish": True,
                "ff_glu": True,
                "ff_no_bias": False,
                "gate_residual": True,
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
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "ce_focal_gamma": 0.,
            "post_emb_norm": False,
            "abs_pos_emb": False,
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
            "opt": schedulefree.AdamWScheduleFree,
            "lr": 5e-4,
        },
        "additional": {
            "epochs": epochs,
            "precision": str(prec) + ", " + torch.get_float32_matmul_precision(),
            "notes": "",
            "comet_exp_key": comet_logger._experiment_key,
        },
    }
    ```