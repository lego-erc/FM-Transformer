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
    XT_QK_NORM_FIX, XT_VERSION, apply_legacy_projection_in_out,
    build_manifold_from_string, migrate_legacy_mult_heads,
    qk_norm_scale_compat, rename_ntokens,
)
from legofmt.data.struct import set_layout
from legofmt.geometry.product_manifold import ProductManifold

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
    base_dist_loss: float
    base_pretrain_batches: int
    base_pretrain_bs: int | None
    pdgid_is_idx: bool
    one_step_euler_fac: float
    one_step_euler_sections: int
    one_step_euler_every: int
    uncert_weighting: bool
    uncert_bins: int
    uncert_min: float
    overflow_delta: float
    edep_overflow_delta: float
    cond_cube: bool
    canon_sym: bool
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


def _build_resolved(
    config: dict,
    model_conf: dict,
    model_args: dict,
    max_seq_l: int,
    pdgids: torch.Tensor,
    state_dict: dict | None,
) -> ResolvedLEGOConfig:
    cond_scalars = tuple(model_conf.get("cond_scalars", ("Density",)))
    n_prefix = len(cond_scalars) + 1  # + edep slot (generated)
    set_layout(cond_scalars)

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

    edep_overflow_delta = model_conf.get("edep_overflow_delta", 0.0)
    if edep_overflow_delta < 0:
        raise ValueError(
            f"edep_overflow_delta must be >= 0, got {edep_overflow_delta}."
        )
    return ResolvedLEGOConfig(
        max_seq_l=max_seq_l,
        pdgids_template=pdgids.contiguous(),
        manifold=build_manifold(model_conf["manifold"]),
        model_args=model_args,
        t_dist=model_conf.get("t_dist", "sd3"),
        t_dist_scale=model_conf.get("t_dist_scale", 1.4),
        t_dist_shift=model_conf.get("t_dist_shift", 1.0),
        ot_coupling=model_conf.get("ot_coupling", False),
        base_dist_loss=model_conf.get("base_dist_loss", 0.0),
        base_pretrain_batches=model_conf.get("base_pretrain_batches", 300),
        base_pretrain_bs=model_conf.get("base_pretrain_bs"),
        pdgid_is_idx=model_conf.get("pdgid_is_idx", False),
        one_step_euler_fac=model_conf.get("one_step_euler_fac", 0.0),
        one_step_euler_sections=sections,
        one_step_euler_every=model_conf.get("one_step_euler_every", 1),
        uncert_weighting=model_conf.get("uncert_weighting", False),
        uncert_bins=model_conf.get("uncert_bins", 16),
        uncert_min=model_conf.get("uncert_min", -6.0),
        overflow_delta=overflow_delta,
        edep_overflow_delta=edep_overflow_delta,
        cond_cube=model_conf.get("cond_cube", False),
        canon_sym=model_conf.get("canon_sym", False),
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


@dataclass(frozen=True)
class ResolvedMultConfig:

    max_seq_len: int
    max_particles: int
    n_ptypes_in: int
    ptypes: torch.Tensor
    ptypes_in: torch.Tensor

    h_dim: int
    in_dim: int
    n_layers: int
    n_heads: int
    dropout: float  # layer_dropout is deliberately not plumbed through: stochastic depth breaks DDP's reducer
    use_abs_pos_emb: bool
    post_emb_norm: bool
    pos_scale: float
    canon_sym: bool
    model_args: dict[str, Any]

    dl_conf: dict
    mm_conf: dict
    opt_conf: dict | None
    config: dict

    state_dict: dict | None

    train_inverse: bool
    inv_model_args: dict[str, Any]
    inv_h_dim: int
    inv_n_layers: int
    inv_n_heads: int


def resolve_mult_config(full_config: dict) -> ResolvedMultConfig:
    full = copy.deepcopy(full_config)
    state_dict = full.get("state_dict")
    config = full.get("config", full)
    if state_dict is None:
        return _resolve_fresh_mult(config)
    return _resolve_from_checkpoint_mult(config, state_dict)


def _resolve_fresh_mult(config: dict) -> ResolvedMultConfig:
    from legofmt.multiplicity.model import MultLoader

    mm_conf = config.setdefault("mm_conf", {})
    dl_conf = config.setdefault("dl_conf", {})
    lds_conf = dl_conf.get("lds_args", {})

    dpath = lds_conf.get("data")
    if dpath is None:
        raise KeyError("Fresh-training mult config requires dl_conf.lds_args.data")

    meta = json.loads(Path(dpath, "meta.json").read_text())
    config.setdefault("additional", {})["data_meta"] = meta
    config["additional"]["x_transformers_version"] = str(XT_VERSION)
    if "max_energy" in meta:  # MultLoader's DataPrep needs it for norm_e
        mm_conf.setdefault("max_energy", meta["max_energy"])
    mm_conf.setdefault("cond_scalars", tuple(meta.get("cond_scalars", ("Density",))))
    set_layout(tuple(mm_conf["cond_scalars"]))  # before MultLoader, which reads layout-dependent accessors
    n_prefix = len(mm_conf["cond_scalars"]) + 1
    mm_conf.setdefault("max_out_particles", meta["ntokens"] - (n_prefix + 1))
    mm_conf.setdefault("ptypes", torch.tensor(meta["particles"]).sort().values)
    mm_conf.setdefault("ptypes_in", torch.tensor(meta["particles_in"]).sort().values)

    if "max_count" not in mm_conf:
        mm_conf["max_count"] = int(MultLoader(config).counts.max().item()) + 1

    return _build_resolved_mult(config, mm_conf, dl_conf, state_dict=None)


def _resolve_from_checkpoint_mult(
    config: dict, state_dict: dict
) -> ResolvedMultConfig:
    mm_conf = config.setdefault("mm_conf", {})
    dl_conf = config.setdefault("dl_conf", {})
    mm_conf.setdefault("max_count", mm_conf.get("max_out_particles"))
    additional = config.setdefault("additional", {})
    model_args = mm_conf.get("model_args", {})
    inv_model_args = mm_conf.get("inv_model_args", model_args)
    for args in {id(d): d for d in (model_args, inv_model_args)}.values():  # may be one dict
        qk_norm_scale_compat(args, additional)
    additional["x_transformers_version"] = str(XT_VERSION)

    ptypes = mm_conf.get("ptypes")
    max_count = mm_conf.get("max_count")
    if ptypes is not None and max_count is not None and len(state_dict) > 0:
        max_seq_len = (
            ptypes.shape[0] if torch.is_tensor(ptypes) else len(ptypes)
        )
        state_dict = migrate_legacy_mult_heads(
            state_dict, max_seq_len=max_seq_len, max_particles=max_count,
        )

    return _build_resolved_mult(config, mm_conf, dl_conf, state_dict=state_dict)


def _build_resolved_mult(
    config: dict,
    mm_conf: dict,
    dl_conf: dict,
    state_dict: dict | None,
) -> ResolvedMultConfig:
    ptypes = mm_conf["ptypes"]
    ptypes_in = mm_conf["ptypes_in"]
    if not torch.is_tensor(ptypes):
        ptypes = torch.tensor(ptypes)
    if not torch.is_tensor(ptypes_in):
        ptypes_in = torch.tensor(ptypes_in)

    cond_scalars = tuple(mm_conf.get("cond_scalars", ("Density",)))
    set_layout(cond_scalars)

    return ResolvedMultConfig(
        max_seq_len=ptypes.shape[0],
        max_particles=mm_conf["max_count"],
        n_ptypes_in=ptypes_in.shape[0],
        ptypes=ptypes.contiguous(),
        ptypes_in=ptypes_in.contiguous(),
        h_dim=mm_conf.get("h_dim", 512),
        in_dim=len(cond_scalars) + 7,
        n_layers=mm_conf.get("n_layers", 6),
        n_heads=mm_conf.get("n_heads", 8),
        dropout=mm_conf.get("dropout", 0.1),
        use_abs_pos_emb=mm_conf.get("use_abs_pos_emb", True),
        post_emb_norm=mm_conf.get("post_emb_norm", True),
        pos_scale=mm_conf.get("pos_scale", 50.0),
        canon_sym=mm_conf.get("canon_sym", False),
        model_args=mm_conf.get("model_args", {}),
        dl_conf=dl_conf,
        mm_conf=mm_conf,
        opt_conf=config.get("opt_conf", mm_conf.get("opt_conf")),  # top-level first, mm_conf for back-compat
        config=config,
        state_dict=state_dict,
        train_inverse=mm_conf.get("train_inverse", False),
        inv_model_args=mm_conf.get("inv_model_args", mm_conf.get("model_args", {})),
        inv_h_dim=mm_conf.get("inv_h_dim", mm_conf.get("h_dim", 512)),
        inv_n_layers=mm_conf.get("inv_n_layers", mm_conf.get("n_layers", 6)),
        inv_n_heads=mm_conf.get("inv_n_heads", mm_conf.get("n_heads", 8)),
    )
