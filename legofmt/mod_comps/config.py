"""Config resolution: raw YAML, or a checkpoint dict, -> a ``Resolved*Config``.

Normalises model args, builds the manifold, and stamps the x-transformers
version into every config. Both entry points deep-copy the dict they are handed,
but ``set_layout`` is global state.
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
        return ProductManifold([_MANIFOLDS[p["name"].lower()]() for p in spec], tuple(p["dim"] for p in spec))
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
    one_step_euler_every: int
    learned_loss_weights: bool
    edep_cell: bool
    edep_whiten: bool            # the E_dep row in units of the base head's per-event prior (BaseDist._edep_whiten)
    edep_whiten_max_sig: float | None  # cap on that unit (logit), so a broad prior keeps a fine MeV grid; None = no cap
    edep_whiten_feed: bool       # hand each event's (mu, unit) to the network on the E_dep row's direction factors
    edep_whiten_global: tuple[float, float] | None  # (c, s): one logit map y = (logit u - c) / s for every event
    dir_zoom: float | None       # theta_s of the outgoing-direction polar zoom (rad); None = off
    dir_zoom_mix: float          # linear share of that zoom; 0 = pure log
    dir_zoom_nucleons_max: int | None  # events with more outgoing nucleons (nuclear interactions) are not zoomed
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

    len_buckets: tuple[int, ...] | None      # training: run each batch / piece at its bucket length
    len_mode: str                            # split | group
    len_split: int
    len_max_tokens: int | None
    gen_len_buckets: tuple[int, ...] | None  # generation (Solvers.forward)

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

    edep_ae: dict | None   # {"config", "state_dict"} of the frozen per-layer E_dep autoencoder
    n_lat: int             # its latent width K, flown as one extra token of the field (0 = off)


def _edep_ae_spec(model_conf: dict) -> dict | None:
    """``model_conf.edep_ae``: ``{path: <scripts/train_edep_ae.py output>}`` is loaded once and replaced by its
    ``{"config", "state_dict"}`` (the path kept), so the saved flow config carries the autoencoder."""
    spec = model_conf.get("edep_ae")
    if spec is not None and "state_dict" not in spec:
        spec = {**torch.load(spec["path"], map_location="cpu", weights_only=False), "path": spec["path"]}
        model_conf["edep_ae"] = spec
    return spec


def _amp_dtype(precision) -> "torch.dtype | None":
    if isinstance(precision, torch.dtype):
        return None if precision is torch.float32 else precision
    head = str(precision).split(",")[0].strip().lower()
    if head.startswith("bf16"):
        return torch.bfloat16
    if head.startswith("16"):
        return torch.float16
    return None


def _unpack(full_config: dict) -> tuple[dict, dict | None]:
    """A deep copy of a checkpoint dict or a bare config, as (config, state_dict or None)."""
    full = copy.deepcopy(full_config)
    return full.get("config", full), full.get("state_dict")


def resolve_legoltng_config(full_config: dict) -> ResolvedLEGOConfig:
    config, state_dict = _unpack(full_config)
    return _resolve_fresh(config) if state_dict is None else _resolve_from_checkpoint(config, state_dict)


def _resolve_fresh(config: dict) -> ResolvedLEGOConfig:
    model_conf = config["model_conf"]
    model_args = model_conf["model_args"]
    dpath = config["dl_conf"]["lds_args"]["data"]
    if dpath.endswith(".pt"):
        raise ValueError(f"Fresh-training path expects a directory containing meta.json; got a .pt file: {dpath}")

    config["dl_conf"].setdefault("data_path", f"{dpath}/data_prepped.pt")

    meta = json.loads(Path(dpath, "meta.json").read_text())
    config.setdefault("additional", {})["data_meta"] = meta
    config["additional"]["x_transformers_version"] = str(XT_VERSION)
    pdgids = torch.tensor(meta["particles"], dtype=torch.int64).sort().values.contiguous()

    max_energy = model_conf.get("max_energy", meta.get("max_energy"))
    if max_energy is None:
        raise KeyError(f"max_energy missing from {dpath}/meta.json and model_conf; "
                       "regenerate meta.json or set model_conf['max_energy'].")
    model_conf["max_energy"] = max_energy
    cutoff_mev = meta.get("cutoff_mev", config["dl_conf"]["lds_args"].get("cutoff_mev"))
    if cutoff_mev is None:
        raise KeyError(f"cutoff_mev missing from {dpath}/meta.json and dl_conf.lds_args.")
    config["dl_conf"]["lds_args"]["cutoff_mev"] = cutoff_mev

    model_args["npdgids"] = pdgids.shape[0] + 1
    model_args.setdefault("max_seq_l", meta["ntokens"])
    model_conf.setdefault("cond_scalars", tuple(meta.get("cond_scalars", ("Density",))))
    if "energy_kin" in meta:
        model_conf.setdefault("energy_kin", bool(meta["energy_kin"]))
    if "cuboid_dim" in meta:
        model_conf.setdefault("cuboid_dim", meta["cuboid_dim"])
    if "edep_excess_max_mev" in meta:   # per incoming pdgid, the data's largest E_dep - E_in: the E_dep cap's margin
        model_conf.setdefault("edep_excess_mev",
                              {int(float(p)): float(m) for p, m in meta["edep_excess_max_mev"].items()})
    elif model_conf.get("edep_wall") and "edep_excess_mev" not in model_conf:   # a dataset older than the key
        from legofmt.data.prep import DataPrep
        set_layout(model_conf["cond_scalars"])
        f = torch.load(config["dl_conf"]["data_path"], map_location="cpu", mmap=True)[0]
        model_conf["edep_excess_mev"] = DataPrep.edep_excess_max(f)
    model_args.setdefault("ntypes", len(model_conf["cond_scalars"]) + 3)
    # ``pdgids`` lives at model_conf scope (one level above model_args) so
    # it is preserved by the manual torch.save round-trip in scripts/train.py.
    model_conf["pdgids"] = pdgids
    return _build_resolved(config, meta["ntokens"], pdgids, state_dict=None)


def _resolve_from_checkpoint(config: dict, state_dict: dict) -> ResolvedLEGOConfig:
    model_conf = config["model_conf"]
    model_args = model_conf["model_args"]
    rename_ntokens(model_args)
    apply_legacy_projection_in_out(model_args, state_dict)
    additional = config.setdefault("additional", {})
    qk_norm_scale_compat(model_args, additional)
    additional["x_transformers_version"] = str(XT_VERSION)
    return _build_resolved(config, model_args["max_seq_l"], model_conf["pdgids"], state_dict=state_dict)


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
    n_cells = 3 + int(model_conf.get("edep_cell", False)) + int(model_conf.get("edep_ae") is not None)  # + lat
    if lv is not None and lv.numel() > n_cells:
        model_conf["loss_weights"]["lv"] = lv.view(-1, n_cells).exp().mean(0).log()
    return model_conf


def resolve_len_buckets(spec, max_seq_l: int) -> tuple[int, ...] | None:
    if spec is None:
        return None
    if spec == "auto":
        spec = [2 ** k for k in range(3, max_seq_l.bit_length())]
    return tuple(sorted({int(b) for b in spec if int(b) < max_seq_l} | {max_seq_l}))


# model_conf keys that ResolvedLEGOConfig carries as they are, with their defaults
_LEGO_DEFAULTS = {
    "t_dist": "sd3", "t_dist_scale": 1.4, "t_dist_shift": 1.0, "ot_coupling": False, "base_pretrain_batches": 300,
    "base_pretrain_bs": None, "pdgid_is_idx": False, "one_step_euler_fac": 0.0, "one_step_euler_sections": 8,
    "one_step_euler_every": 1, "learned_loss_weights": False, "edep_cell": False, "edep_whiten": False,
    "edep_whiten_max_sig": None, "edep_whiten_feed": False, "dir_zoom": None, "dir_zoom_mix": 0.0,
    "dir_zoom_nucleons_max": None, "max_loss_weight": -6.0, "overflow_delta": 0.0, "canon_sym": False,
    "sym_aug": False, "reflow_path": None, "reflow_every": 1, "reflow_start_epoch": 0,
}


def _build_resolved(config: dict, max_seq_l: int, pdgids: torch.Tensor, state_dict: dict | None) -> ResolvedLEGOConfig:
    model_conf, dl_conf = config["model_conf"], config["dl_conf"]
    model_args = model_conf["model_args"]
    migrate_loss_weight_keys(model_conf)
    c = {k: model_conf.get(k, d) for k, d in _LEGO_DEFAULTS.items()}
    cond_scalars = tuple(model_conf.get("cond_scalars", ("Density",)))
    n_prefix = len(cond_scalars) + 1  # + edep slot (generated)
    set_layout(cond_scalars)
    geom = model_args.get("geom_encoder")
    if geom is not None:
        layers = [[float(v) for v in row] for row in geom.get("layers") or []] * int(geom.pop("repeat", 1))
        if not layers or any(len(r) != 4 or min(r) <= 0 for r in layers):
            raise ValueError("model_args.geom_encoder.layers must be rows of positive [thickness mm, density, Z, A].")
        geom["layers"] = layers
        geom.setdefault("hide_scalars", True)
        geom["n_prefix"] = n_prefix

    if not state_dict and c["one_step_euler_fac"] > 0:  # loading weights never trains
        if not model_args.get("step_cond", False):
            raise ValueError("one_step_euler_fac > 0 requires model_args.step_cond=True; without it the network never "
                             "sees the step size and the loss collapses to a curvature penalty on the velocity field.")
        if c["one_step_euler_sections"] < 1:
            raise ValueError(f"one_step_euler_sections must be >= 1, got {c['one_step_euler_sections']}.")
    if c["dir_zoom"] is not None and not c["dir_zoom"] > 0:
        raise ValueError(f"dir_zoom must be > 0 (rad) or absent, got {c['dir_zoom']}.")
    if not 0.0 <= c["dir_zoom_mix"] < 1.0:
        raise ValueError(f"dir_zoom_mix must be in [0, 1), got {c['dir_zoom_mix']}.")
    nuc = c["dir_zoom_nucleons_max"]
    if nuc is not None and not (isinstance(nuc, int) and nuc >= 0):
        raise ValueError(f"dir_zoom_nucleons_max must be an int >= 0 or absent, got {nuc}.")
    if c["overflow_delta"] < 0:
        raise ValueError(f"overflow_delta must be >= 0, got {c['overflow_delta']}.")
    cuboid_dim = model_conf.get("cuboid_dim")
    if c["canon_sym"] and cuboid_dim and len(set(cuboid_dim)) > 1:
        raise ValueError(f"canon_sym rotates every face onto +x, which is a symmetry of a cube only; "
                         f"got cuboid_dim={cuboid_dim}.")
    len_buckets = resolve_len_buckets(dl_conf.get("len_buckets"), max_seq_l)
    len_mode = dl_conf.get("len_mode", "split")
    len_split = dl_conf.get("len_split", 8)
    if len_buckets is not None:
        if len_mode not in ("split", "group"):
            raise ValueError(f"dl_conf.len_mode must be 'split' or 'group', got {len_mode!r}.")
        if len_mode == "split" and dl_conf["bs"] % len_split:
            raise ValueError(f"dl_conf.len_split={len_split} must divide dl_conf.bs={dl_conf['bs']}.")
    edep_ae = _edep_ae_spec(model_conf)
    if edep_ae is not None:
        # len_mode group keeps a batch whole and in order (the per-layer targets stay with their events); split
        # reorders and runs pieces, which do not carry the latent token
        if ((len_buckets is not None and len_mode != "group") or c["reflow_path"] or c["reflow_start_epoch"] > 0
                or (model_conf.get("mask_conf") or {}).get("p_forward", 1.0) < 1):
            raise ValueError("model_conf.edep_ae does not support training len_buckets in split mode, reflow or "
                             "inverse-mask events (mask_conf.p_forward < 1) yet.")
        model_args.setdefault("lat_dim", edep_ae["config"]["K"])   # the field's latent token (CFMTrafo_x)

    whiten_global = model_conf.get("edep_whiten_global")
    return ResolvedLEGOConfig(
        **c,
        max_seq_l=max_seq_l,
        pdgids_template=pdgids.contiguous(),
        manifold=build_manifold(model_conf["manifold"]),
        model_args=model_args,
        edep_whiten_global=None if whiten_global is None else tuple(whiten_global),
        max_loss_weight_flow=model_conf.get("max_loss_weight_flow", c["max_loss_weight"]),
        max_loss_weight_edep=model_conf.get("max_loss_weight_edep", c["max_loss_weight"]),
        cond_scalars=cond_scalars,
        n_prefix=n_prefix,
        mask_conf=model_conf.get("mask_conf", {}),
        max_energy=model_conf["max_energy"],
        cutoff_mev=dl_conf["lds_args"]["cutoff_mev"],
        amp_dtype=_amp_dtype((config.get("additional") or {}).get("precision")),
        len_buckets=len_buckets,
        len_mode=len_mode,
        len_split=len_split,
        len_max_tokens=dl_conf.get("len_max_tokens"),
        gen_len_buckets=resolve_len_buckets(config.get("odeint_conf", {}).get("len_buckets"), max_seq_l),
        dl_conf=dl_conf,
        opt_conf=config["opt_conf"],
        odeint_conf=config.get("odeint_conf", {}),
        val_conf=config.get("val_conf", {}),
        config=config,
        state_dict=state_dict,
        reflow_kwargs=model_conf.get("reflow_kwargs", {}),
        edep_ae=edep_ae,
        n_lat=edep_ae["config"]["K"] if edep_ae else 0,
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
    sym_aug: bool
    model_args: dict[str, Any]

    dl_conf: dict
    mm_conf: dict
    opt_conf: dict | None
    val_conf: dict
    config: dict

    state_dict: dict | None

    train_inverse: bool
    inv_model_args: dict[str, Any]
    inv_h_dim: int
    inv_n_layers: int
    inv_n_heads: int


def resolve_mult_config(full_config: dict) -> ResolvedMultConfig:
    config, state_dict = _unpack(full_config)
    return _resolve_fresh_mult(config) if state_dict is None else _resolve_from_checkpoint_mult(config, state_dict)


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
    lds_conf["cutoff_mev"] = meta.get("cutoff_mev", lds_conf.get("cutoff_mev"))  # as the flow: the file's cut wins
    if "bs" in mm_conf and dl_conf.get("bs", mm_conf["bs"]) != mm_conf["bs"]:
        raise ValueError(f"mm_conf.bs={mm_conf['bs']} (the loader's) != dl_conf.bs={dl_conf['bs']}: set one")
    mm_conf.setdefault("cond_scalars", tuple(meta.get("cond_scalars", ("Density",))))
    set_layout(tuple(mm_conf["cond_scalars"]))  # before MultLoader, which reads layout-dependent accessors
    n_prefix = len(mm_conf["cond_scalars"]) + 1
    mm_conf.setdefault("max_out_particles", meta["ntokens"] - (n_prefix + 1))
    mm_conf.setdefault("ptypes", torch.tensor(meta["particles"]).sort().values)
    mm_conf.setdefault("ptypes_in", torch.tensor(meta["particles_in"]).sort().values)

    if "max_count" not in mm_conf:
        mm_conf["max_count"] = int(MultLoader(config).counts.max().item()) + 1
    return _build_resolved_mult(config, state_dict=None)


def _resolve_from_checkpoint_mult(config: dict, state_dict: dict) -> ResolvedMultConfig:
    mm_conf = config.setdefault("mm_conf", {})
    config.setdefault("dl_conf", {})
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
        state_dict = migrate_legacy_mult_heads(state_dict, max_seq_len=len(ptypes), max_particles=max_count)
    return _build_resolved_mult(config, state_dict=state_dict)


# mm_conf keys that ResolvedMultConfig carries as they are, with their defaults
_MULT_DEFAULTS = {
    "h_dim": 512, "n_layers": 6, "n_heads": 8, "dropout": 0.1, "use_abs_pos_emb": True, "post_emb_norm": True,
    "pos_scale": 50.0, "canon_sym": False, "sym_aug": False, "train_inverse": False,
}


def _build_resolved_mult(config: dict, state_dict: dict | None) -> ResolvedMultConfig:
    mm_conf = config["mm_conf"]
    c = {k: mm_conf.get(k, d) for k, d in _MULT_DEFAULTS.items()}
    ptypes, ptypes_in = torch.as_tensor(mm_conf["ptypes"]), torch.as_tensor(mm_conf["ptypes_in"])
    cond_scalars = tuple(mm_conf.get("cond_scalars", ("Density",)))
    set_layout(cond_scalars)

    return ResolvedMultConfig(
        **c,
        max_seq_len=ptypes.shape[0],
        max_particles=mm_conf["max_count"],
        n_ptypes_in=ptypes_in.shape[0],
        ptypes=ptypes.contiguous(),
        ptypes_in=ptypes_in.contiguous(),
        in_dim=len(cond_scalars) + 7,
        model_args=mm_conf.get("model_args", {}),
        dl_conf=config["dl_conf"],
        mm_conf=mm_conf,
        opt_conf=config.get("opt_conf", mm_conf.get("opt_conf")),  # top-level first, mm_conf for back-compat
        val_conf=config.get("val_conf", {}),
        config=config,
        state_dict=state_dict,
        inv_model_args=mm_conf.get("inv_model_args", mm_conf.get("model_args", {})),
        inv_h_dim=mm_conf.get("inv_h_dim", c["h_dim"]),
        inv_n_layers=mm_conf.get("inv_n_layers", c["n_layers"]),
        inv_n_heads=mm_conf.get("inv_n_heads", c["n_heads"]),
    )
