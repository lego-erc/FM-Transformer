"""Config resolution: raw YAML, or a checkpoint dict, -> a ``Resolved*Config``.

Normalises model args, builds the manifold, and stamps the x-transformers
version into every config. Both entry points mutate the dict they are handed
(via ``set_layout`` and ``setdefault``), and ``set_layout`` is global state.
Support for older artefacts lives in ``legofmt.compat``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.compat import (
    XT_VERSION, apply_legacy_projection_in_out,
    build_manifold_from_string,
    qk_norm_scale_compat, rename_ntokens,
)
from legofmt.data.struct import set_layout
from legofmt.geometry.product_manifold import ProductManifold

_NEUTRAL_PDGIDS = (22, 2112, 130, 310, 12, -12, 14, -14, 3122)

_MANIFOLDS: dict[str, type] = {
    "euclidean": Euclidean,
    "sphere": Sphere,
}


def build_manifold(spec: str | list) -> ProductManifold:
    if isinstance(spec, str):
        return build_manifold_from_string(spec)

    if isinstance(spec, list):
        manifolds = [_MANIFOLDS[p["name"].lower()]() for p in spec]
        dims = tuple(p["dim"] for p in spec)
        return ProductManifold(manifolds, dims)

    raise ValueError(f"Cannot build manifold from spec: {spec!r}")


@dataclass(frozen=True)
class ResolvedLEGOConfig:

    max_seq_l: int
    pdgids_template: torch.Tensor
    manifold: ProductManifold
    model_args: dict[str, Any]

    t_dist: str
    t_dist_scale: float
    t_dist_shift: float
    ot_coupling: bool
    base_pretrain_batches: int
    base_pretrain_bs: int | None
    pdgid_is_idx: bool
    one_step_euler_fac: float
    one_step_euler_sections: int
    species_fac: float
    passthrough_fac: float
    passthrough_pdgids: tuple[int, ...]
    one_step_euler_every: int
    learned_loss_weights: bool
    edep_cell: bool
    max_loss_weight: float       # floors lv, so weight_bound = e^-max_loss_weight
    max_loss_weight_flow: float  # the same bound for the flow-map's own cell
    max_loss_weight_edep: float
    overflow_delta: float
    canon_sym: bool
    sym_aug: bool
    cond_scalars: tuple[str, ...]
    n_prefix: int

    mask_conf: dict

    max_energy: float
    cutoff_mev: float
    amp_dtype: torch.dtype | None

    dl_conf: dict
    opt_conf: dict
    odeint_conf: dict
    val_conf: dict
    config: dict

    state_dict: dict | None

    reflow_path: str | None
    reflow_kwargs: dict
    reflow_every: int
    reflow_start_epoch: int


def _amp_dtype(precision) -> "torch.dtype | None":
    if isinstance(precision, torch.dtype):
        return None if precision is torch.float32 else precision
    head = str(precision).split(",")[0].strip().lower()
    if head.startswith("bf16"):
        return torch.bfloat16
    if head.startswith("16"):
        return torch.float16
    return None


def resolve_legoltng_config(full_config: dict) -> ResolvedLEGOConfig:
    full = copy.deepcopy(full_config)
    state_dict = full.get("state_dict")
    config = full.get("config", full)

    if state_dict is None:
        return _resolve_fresh(config)
    return _resolve_from_checkpoint(config, state_dict)


def _resolve_fresh(config: dict) -> ResolvedLEGOConfig:
    model_conf = config["model_conf"]
    model_args = model_conf["model_args"]
    dpath = config["dl_conf"]["lds_args"]["data"]

    if dpath.endswith(".pt"):
        raise ValueError(
            "Fresh-training path expects a directory containing meta.json; "
            f"got a .pt file: {dpath}"
        )

    config["dl_conf"].setdefault("data_path", f"{dpath}/data_prepped.pt")

    meta = json.loads(Path(dpath, "meta.json").read_text())
    config.setdefault("additional", {})["data_meta"] = meta
    config["additional"]["x_transformers_version"] = str(XT_VERSION)
    max_seq_l = meta["ntokens"]
    pdgids = (
        torch.tensor(meta["particles"], dtype=torch.int64).sort().values.contiguous()
    )

    max_energy = model_conf.get("max_energy", meta.get("max_energy"))
    if max_energy is None:
        raise KeyError(
            f"max_energy missing from {dpath}/meta.json and model_conf; "
            "regenerate meta.json or set model_conf['max_energy']."
        )
    model_conf["max_energy"] = max_energy
    cutoff_mev = meta.get(
        "cutoff_mev", config["dl_conf"]["lds_args"].get("cutoff_mev")
    )
    if cutoff_mev is None:
        raise KeyError(
            f"cutoff_mev missing from {dpath}/meta.json and dl_conf.lds_args."
        )
    config["dl_conf"]["lds_args"]["cutoff_mev"] = cutoff_mev

    model_args["npdgids"] = pdgids.shape[0] + 1
    model_args.setdefault("max_seq_l", max_seq_l)
    model_conf.setdefault("cond_scalars", tuple(meta.get("cond_scalars", ("Density",))))
    if "energy_kin" in meta:
        model_conf.setdefault("energy_kin", bool(meta["energy_kin"]))
    if "cuboid_dim" in meta:
        model_conf.setdefault("cuboid_dim", meta["cuboid_dim"])
    model_args.setdefault("ntypes", len(model_conf["cond_scalars"]) + 3)
    # ``pdgids`` lives at model_conf scope (one level above model_args) so
    # it is preserved by the manual torch.save round-trip in scripts/train.py.
    model_conf["pdgids"] = pdgids

    return _build_resolved(
        config, model_conf, model_args, max_seq_l, pdgids, state_dict=None,
    )


def _resolve_from_checkpoint(config: dict, state_dict: dict) -> ResolvedLEGOConfig:
    model_conf = config["model_conf"]
    model_args = model_conf["model_args"]

    rename_ntokens(model_args)
    apply_legacy_projection_in_out(model_args, state_dict)
    additional = config.setdefault("additional", {})
    qk_norm_scale_compat(model_args, additional)
    additional["x_transformers_version"] = str(XT_VERSION)

    return _build_resolved(
        config, model_conf, model_args,
        model_args["max_seq_l"], model_conf["pdgids"],
        state_dict=state_dict,
    )


# --- back-compat: the 2026-09 loss-weighting rename and the removal of the t-bins ---
# `uncert_bins` 16 vs 1 measured as a null on generated E_dep (paired dW1 +0.015 +- 0.124
# MeV against a 0.75 seed sd), so the per-t-bin table is gone and `lv` is one cell per
# channel. A saved `lv` of size `bins*3` collapses through the mean of its variances,
# because a converged Kendall cell sits at `lv = log(L)`.
_RENAMED_KEYS = {
    "uncert_weighting": "learned_loss_weights",
    "uncert_min": "max_loss_weight",
    "uncert_min_flow": "max_loss_weight_flow",
    "uncert_lv": "loss_weights",
}


def migrate_loss_weight_keys(model_conf: dict) -> dict:
    """Rewrite the pre-2026-09 ``uncert_*`` keys and drop the removed t-bin axis."""
    for old, new in _RENAMED_KEYS.items():
        if old in model_conf:
            model_conf.setdefault(new, model_conf.pop(old))
    model_conf.pop("uncert_bins", None)
    lv = (model_conf.get("loss_weights") or {}).get("lv")
    n_cells = 3 + int(model_conf.get("edep_cell", False))
    if lv is not None and lv.numel() > n_cells:
        model_conf["loss_weights"]["lv"] = lv.view(-1, n_cells).exp().mean(0).log()
    return model_conf


def _build_resolved(
    config: dict,
    model_conf: dict,
    model_args: dict,
    max_seq_l: int,
    pdgids: torch.Tensor,
    state_dict: dict | None,
) -> ResolvedLEGOConfig:
    migrate_loss_weight_keys(model_conf)
    cond_scalars = tuple(model_conf.get("cond_scalars", ("Density",)))
    n_prefix = len(cond_scalars) + 1  # + edep slot (generated)
    set_layout(cond_scalars)
    model_args.setdefault("n_cond", len(cond_scalars))
    model_args.setdefault("pt_dim", model_conf.get("passthrough_dim", 64))

    sections = model_conf.get("one_step_euler_sections", 8)
    if model_conf.get("one_step_euler_fac", 0.0) > 0:
        if not model_args.get("step_cond", False):
            raise ValueError(
                "one_step_euler_fac > 0 requires model_args.step_cond=True; "
                "without it the network never sees the step size and the loss "
                "collapses to a curvature penalty on the velocity field."
            )
        if sections < 1:
            raise ValueError(
                f"one_step_euler_sections must be >= 1, got {sections}."
            )
    overflow_delta = model_conf.get("overflow_delta", 0.0)
    if overflow_delta < 0:
        raise ValueError(f"overflow_delta must be >= 0, got {overflow_delta}.")
    cuboid_dim = model_conf.get("cuboid_dim")
    if model_conf.get("canon_sym", False) and cuboid_dim and len(set(cuboid_dim)) > 1:
        raise ValueError(
            f"canon_sym rotates every face onto +x, which is a symmetry of a cube only; "
            f"got cuboid_dim={cuboid_dim}."
        )
    max_loss_weight = model_conf.get("max_loss_weight", -6.0)

    return ResolvedLEGOConfig(
        max_seq_l=max_seq_l,
        pdgids_template=pdgids.contiguous(),
        manifold=build_manifold(model_conf["manifold"]),
        model_args=model_args,
        t_dist=model_conf.get("t_dist", "sd3"),
        t_dist_scale=model_conf.get("t_dist_scale", 1.4),
        t_dist_shift=model_conf.get("t_dist_shift", 1.0),
        ot_coupling=model_conf.get("ot_coupling", False),
        base_pretrain_batches=model_conf.get("base_pretrain_batches", 300),
        base_pretrain_bs=model_conf.get("base_pretrain_bs"),
        pdgid_is_idx=model_conf.get("pdgid_is_idx", False),
        one_step_euler_fac=model_conf.get("one_step_euler_fac", 0.0),
        one_step_euler_sections=sections,
        species_fac=model_conf.get("species_fac", 1.0),
        passthrough_fac=model_conf.get("passthrough_fac", 1.0),
        passthrough_pdgids=tuple(model_conf.get("passthrough_pdgids", _NEUTRAL_PDGIDS)),
        one_step_euler_every=model_conf.get("one_step_euler_every", 1),
        learned_loss_weights=model_conf.get("learned_loss_weights", False),
        edep_cell=model_conf.get("edep_cell", False),
        max_loss_weight=max_loss_weight,
        max_loss_weight_flow=model_conf.get("max_loss_weight_flow", max_loss_weight),
        max_loss_weight_edep=model_conf.get("max_loss_weight_edep", max_loss_weight),
        overflow_delta=overflow_delta,
        canon_sym=model_conf.get("canon_sym", False),
        sym_aug=model_conf.get("sym_aug", False),
        cond_scalars=cond_scalars,
        n_prefix=n_prefix,
        mask_conf=model_conf.get("mask_conf", {}),
        max_energy=model_conf["max_energy"],
        cutoff_mev=config["dl_conf"]["lds_args"]["cutoff_mev"],
        amp_dtype=_amp_dtype((config.get("additional") or {}).get("precision")),
        dl_conf=config["dl_conf"],
        opt_conf=config["opt_conf"],
        odeint_conf=config.get("odeint_conf", {}),
        val_conf=config.get("val_conf", {}),
        config=config,
        state_dict=state_dict,
        reflow_path=model_conf.get("reflow_path"),
        reflow_kwargs=model_conf.get("reflow_kwargs", {}),
        reflow_every=model_conf.get("reflow_every", 1),
        reflow_start_epoch=model_conf.get("reflow_start_epoch", 0),
    )
