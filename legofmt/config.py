"""Config resolution: raw YAML, or a checkpoint dict, -> a ``Resolved*Config``.

The two ``Resolved*Config`` classes are the table of what a config resolves to: a field with a default is a config key
carried as it is, the default the value it takes when absent; every other field is derived, its comment naming the
source. Then come the resolvers, which read the dataset's meta.json (fresh training) or the checkpoint, build the
manifold and stamp the x-transformers version into every config; the checks a config must pass are at the bottom.
Both entry points deep-copy the dict they are handed, but ``set_layout`` is global state. Support for older artefacts,
old config keys among them, lives in ``legofmt.compat``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.compat import (
    XT_VERSION, apply_legacy_projection_in_out,
    build_manifold_from_string, edep_excess_from_data, migrate_legacy_mult_heads,
    migrate_loss_weight_keys, qk_norm_scale_compat, rename_ntokens,
)
from legofmt.data.struct import set_layout
from legofmt.geometry.product_manifold import ProductManifold


@dataclass(frozen=True, kw_only=True)
class ResolvedLEGOConfig:
    """A flow config, resolved.

    A field with a default is a ``model_conf`` key, carried as it is; the others are derived in ``_build_resolved``.
    """

    # ---- model_conf keys, as they are ----------------------------------------------------------------------------
    t_dist: str = "sd3"
    t_dist_scale: float = 1.4
    t_dist_shift: float = 1.0
    ot_coupling: bool = False
    base_pretrain_batches: int = 300
    base_pretrain_bs: int | None = None
    pdgid_is_idx: bool = False
    one_step_euler_fac: float = 0.0
    one_step_euler_sections: int = 8
    one_step_euler_every: int = 1
    learned_loss_weights: bool = False
    edep_cell: bool = False
    edep_whiten: bool = False                  # the E_dep token in units of the base head's per-event prior;
                                               # off: legacy.edep_token
    edep_whiten_max_sig: float | None = None   # legacy.edep_token: cap on that unit (logit) for a finer MeV grid
    edep_whiten_feed: bool = False             # legacy.edep_token: each event's (mu, unit) on the E_dep token's
                                               # direction factors
    dir_zoom: float | None = None              # theta_s of the outgoing-direction polar zoom (rad); None = off
    dir_zoom_mix: float = 0.0                  # linear share of that zoom; 0 = pure log
    dir_zoom_nucleons_max: int | None = None   # events with more outgoing nucleons (nuclear interactions) are
                                               # not zoomed
    max_loss_weight: float = -6.0              # floors lv, so weight_bound = e^-max_loss_weight
    overflow_delta: float = 0.0
    canon_sym: bool = False
    sym_aug: bool = False
    reflow_path: str | None = None
    reflow_every: int = 1
    reflow_start_epoch: int = 0

    # ---- derived -------------------------------------------------------------------------------------------------
    max_seq_l: int                     # model_args.max_seq_l; fresh: meta.json ntokens
    pdgids_template: torch.Tensor      # model_conf.pdgids; fresh: meta.json particles, sorted
    manifold: ProductManifold          # build_manifold(model_conf.manifold)
    model_args: dict[str, Any]         # model_conf.model_args, completed by the resolvers
    cond_scalars: tuple[str, ...]      # model_conf.cond_scalars (default ("Density",)); fresh: else meta.json
    n_prefix: int                      # len(cond_scalars) + 1: the generated E_dep's slot
    max_loss_weight_flow: float        # model_conf key, default max_loss_weight: the flow-map's own cell
    max_loss_weight_edep: float        # model_conf key, default max_loss_weight
    edep_whiten_global: tuple[float, float] | None  # model_conf key; legacy.edep_token: (c, s), one map
                                                    # y = (logit u - c) / s for all
    mask_conf: dict                    # model_conf key, default {}
    inverse_events: bool               # mask_conf.p_forward < 1: some events generate the incoming particle (legacy)
    max_energy: float                  # model_conf.max_energy; fresh: else meta.json
    cutoff_mev: float                  # dl_conf.lds_args.cutoff_mev; fresh: meta.json first
    amp_dtype: torch.dtype | None      # additional.precision: bf16* -> bfloat16, 16* -> float16, else None
    len_buckets: tuple[int, ...] | None      # dl_conf.len_buckets: training runs each batch / piece at its bucket
                                             # length
    len_mode: str                            # dl_conf key, default "split" (| "group")
    len_split: int                           # dl_conf key, default 8
    len_max_tokens: int | None               # dl_conf key
    reflow_kwargs: dict                # model_conf key, default {}
    edep_ae: dict | None               # model_conf.edep_ae, loaded: {"config", "state_dict"} of the frozen per-layer
                                       # E_dep autoencoder
    n_lat: int                         # its latent width K, flown as one extra token of the field (0 = off)
    dl_conf: dict
    opt_conf: dict
    odeint_conf: dict                  # default {}
    val_conf: dict                     # default {}
    config: dict                       # the whole config, as resolved
    state_dict: dict | None            # the checkpoint's; None: fresh training


def resolve_legoltng_config(full_config: dict) -> ResolvedLEGOConfig:
    config, state_dict = _unpack(full_config)
    return _resolve_fresh(config) if state_dict is None else _resolve_from_checkpoint(config, state_dict)


def _resolve_fresh(config: dict) -> ResolvedLEGOConfig:
    """Resolves a config for fresh training: the dataset's meta.json fills in what the config leaves out."""
    model_conf, lds_args = config["model_conf"], config["dl_conf"]["lds_args"]
    model_args = model_conf["model_args"]
    meta = _read_meta(config)
    pdgids = torch.tensor(meta["particles"], dtype=torch.int64).sort().values.contiguous()

    model_conf["max_energy"] = model_conf.get("max_energy", meta.get("max_energy"))   # the config's first
    lds_args["cutoff_mev"] = meta.get("cutoff_mev", lds_args.get("cutoff_mev"))       # the file's first
    model_args["npdgids"] = pdgids.shape[0] + 1                                       # + the padding's
    model_args.setdefault("max_seq_l", meta["ntokens"])
    model_conf.setdefault("cond_scalars", tuple(meta.get("cond_scalars", ("Density",))))
    if "energy_kin" in meta:
        model_conf.setdefault("energy_kin", bool(meta["energy_kin"]))
    if "cuboid_dim" in meta:
        model_conf.setdefault("cuboid_dim", meta["cuboid_dim"])
    if "edep_excess_max_mev" in meta:   # per incoming pdgid, the data's largest E_dep - E_in: the E_dep cap's margin
        model_conf.setdefault("edep_excess_mev",
                              {int(float(p)): float(m) for p, m in meta["edep_excess_max_mev"].items()})
    else:
        edep_excess_from_data(config)   # a dataset older than the key
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


def _build_resolved(config: dict, max_seq_l: int, pdgids: torch.Tensor, state_dict: dict | None) -> ResolvedLEGOConfig:
    model_conf, dl_conf = config["model_conf"], config["dl_conf"]
    model_args = model_conf["model_args"]
    migrate_loss_weight_keys(model_conf)
    c = {f.name: model_conf.get(f.name, f.default) for f in fields(ResolvedLEGOConfig) if f.default is not MISSING}
    cond_scalars = tuple(model_conf.get("cond_scalars", ("Density",)))
    n_prefix = len(cond_scalars) + 1  # + edep slot (generated)
    set_layout(cond_scalars)
    _expand_geom_encoder(model_args.get("geom_encoder"), n_prefix)
    if model_args.get("cond_pool"):
        model_args["cond_pool"] = n_prefix   # the field takes the number of prefix tokens (CFMTrafo_x)
    len_buckets = resolve_len_buckets(dl_conf.get("len_buckets"), max_seq_l)
    _check_lego(config, c, cond_scalars, len_buckets, state_dict)
    edep_ae = _edep_ae_spec(model_conf)
    if edep_ae is not None:
        model_args.setdefault("lat_dim", edep_ae["config"]["K"])   # the field's latent token (CFMTrafo_x)

    mask_conf, whiten_global = model_conf.get("mask_conf"), model_conf.get("edep_whiten_global")
    return ResolvedLEGOConfig(
        **c,
        max_seq_l=max_seq_l,
        pdgids_template=pdgids.contiguous(),
        manifold=build_manifold(model_conf["manifold"]),
        model_args=model_args,
        cond_scalars=cond_scalars,
        n_prefix=n_prefix,
        max_loss_weight_flow=model_conf.get("max_loss_weight_flow", c["max_loss_weight"]),
        max_loss_weight_edep=model_conf.get("max_loss_weight_edep", c["max_loss_weight"]),
        edep_whiten_global=None if whiten_global is None else tuple(whiten_global),
        mask_conf=model_conf.get("mask_conf", {}),
        inverse_events=bool(mask_conf) and mask_conf.get("p_forward", 0.5) < 1,
        max_energy=model_conf["max_energy"],
        cutoff_mev=dl_conf["lds_args"]["cutoff_mev"],
        amp_dtype=amp_dtype((config.get("additional") or {}).get("precision")),
        len_buckets=len_buckets,
        len_mode=dl_conf.get("len_mode", "split"),
        len_split=dl_conf.get("len_split", 8),
        len_max_tokens=dl_conf.get("len_max_tokens"),
        reflow_kwargs=model_conf.get("reflow_kwargs", {}),
        edep_ae=edep_ae,
        n_lat=edep_ae["config"]["K"] if edep_ae else 0,
        dl_conf=dl_conf,
        opt_conf=config["opt_conf"],
        odeint_conf=config.get("odeint_conf", {}),
        val_conf=config.get("val_conf", {}),
        config=config,
        state_dict=state_dict,
    )


@dataclass(frozen=True, kw_only=True)
class ResolvedMultConfig:
    """A multiplicity config, resolved.

    A field with a default is an ``mm_conf`` key, carried as it is; the others are derived in ``_build_resolved_mult``.
    """

    # ---- mm_conf keys, as they are -------------------------------------------------------------------------------
    h_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1   # layer_dropout is deliberately not plumbed through: stochastic depth breaks DDP's reducer
    use_abs_pos_emb: bool = True
    post_emb_norm: bool = True
    canon_sym: bool = False
    sym_aug: bool = False
    train_inverse: bool = False

    # ---- derived -------------------------------------------------------------------------------------------------
    max_seq_len: int                   # len(ptypes)
    max_particles: int                 # mm_conf.max_count; fresh: else the data's largest count + 1
    n_ptypes_in: int                   # len(ptypes_in)
    ptypes: torch.Tensor               # mm_conf.ptypes; fresh: else meta.json particles, sorted
    ptypes_in: torch.Tensor            # mm_conf.ptypes_in; fresh: else meta.json particles_in, sorted
    in_dim: int                        # len(cond_scalars) + 7
    pos_scale: torch.Tensor            # mm_conf.pos_scale (default 50.0) as float32: half-width(s) d of
                                       # GeomTrafos.to_cuboid, 0-dim (a cube) or per axis (3,)
    model_args: dict[str, Any]         # mm_conf key, default {}
    dl_conf: dict
    mm_conf: dict
    opt_conf: dict | None              # top-level opt_conf, else mm_conf.opt_conf (back-compat)
    val_conf: dict                     # default {}
    config: dict                       # the whole config, as resolved
    state_dict: dict | None            # the checkpoint's; None: fresh training
    data: Any                          # the dataset (MultLoader), when resolving read it for its largest count
    inv_model_args: dict[str, Any]     # mm_conf key, default model_args
    inv_h_dim: int                     # mm_conf key, default h_dim
    inv_n_layers: int                  # mm_conf key, default n_layers
    inv_n_heads: int                   # mm_conf key, default n_heads


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
    mm_conf.setdefault("cond_scalars", tuple(meta.get("cond_scalars", ("Density",))))
    _check_mult(mm_conf, dl_conf, meta)
    set_layout(tuple(mm_conf["cond_scalars"]))  # before MultLoader, which reads layout-dependent accessors
    n_prefix = len(mm_conf["cond_scalars"]) + 1
    mm_conf.setdefault("max_out_particles", meta["ntokens"] - (n_prefix + 1))
    mm_conf.setdefault("ptypes", torch.tensor(meta["particles"]).sort().values)
    mm_conf.setdefault("ptypes_in", torch.tensor(meta["particles_in"]).sort().values)

    data = None
    if "max_count" not in mm_conf:   # the count heads' size: the dataset is read here, and kept for training
        data = MultLoader(config)
        mm_conf["max_count"] = int(data.counts.max()) + 1
    return _build_resolved_mult(config, state_dict=None, data=data)


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
        state_dict = migrate_legacy_mult_heads(state_dict, max_seq_len=len(ptypes))
    return _build_resolved_mult(config, state_dict=state_dict)


def _build_resolved_mult(config: dict, state_dict: dict | None, data=None) -> ResolvedMultConfig:
    mm_conf = config["mm_conf"]
    c = {f.name: mm_conf.get(f.name, f.default) for f in fields(ResolvedMultConfig) if f.default is not MISSING}
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
        pos_scale=torch.as_tensor(mm_conf.get("pos_scale", 50.0), dtype=torch.float32),
        model_args=mm_conf.get("model_args", {}),
        dl_conf=config["dl_conf"],
        mm_conf=mm_conf,
        opt_conf=config.get("opt_conf", mm_conf.get("opt_conf")),  # top-level first, mm_conf for back-compat
        val_conf=config.get("val_conf", {}),
        config=config,
        state_dict=state_dict,
        data=data,
        inv_model_args=mm_conf.get("inv_model_args", mm_conf.get("model_args", {})),
        inv_h_dim=mm_conf.get("inv_h_dim", c["h_dim"]),
        inv_n_layers=mm_conf.get("inv_n_layers", c["n_layers"]),
        inv_n_heads=mm_conf.get("inv_n_heads", c["n_heads"]),
    )


# ---- plumbing --------------------------------------------------------------------------------------------------------

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


def resolve_len_buckets(spec, max_seq_l: int) -> tuple[int, ...] | None:
    """Returns the ascending bucket lengths a ``len_buckets`` spec names, ``max_seq_l`` the last; None: no buckets.

    ``"auto"``: four per octave from 8 on (8, 10, 12, 14, 16, 20, ...). Padded to these, 7 % of the tokens a training
    step runs are padding (Ar cube, 16M events); to powers of two it is 29 %, to ``max_seq_l`` 85 %.
    """
    if spec is None:
        return None
    if spec == "auto":
        spec = [2 ** k * j // 4 for k in range(3, max_seq_l.bit_length()) for j in (4, 5, 6, 7)]
    return tuple(sorted({int(b) for b in spec if int(b) < max_seq_l} | {max_seq_l}))


def _unpack(full_config: dict) -> tuple[dict, dict | None]:
    """Returns a deep copy of a checkpoint dict or a bare config, as (config, state_dict or None)."""
    full = copy.deepcopy(full_config)
    return full.get("config", full), full.get("state_dict")


def _read_meta(config: dict) -> dict:
    """Reads the dataset directory's meta.json, and stamps it and the x-transformers version into ``additional``.

    Raises:
        ValueError: ``dl_conf.lds_args.data`` is a ``.pt`` file, not a directory.
    """
    dpath = config["dl_conf"]["lds_args"]["data"]
    if dpath.endswith(".pt"):
        raise ValueError(f"Fresh-training path expects a directory containing meta.json; got a .pt file: {dpath}")
    config["dl_conf"].setdefault("data_path", f"{dpath}/data_prepped.pt")
    meta = json.loads(Path(dpath, "meta.json").read_text())
    config.setdefault("additional", {})["data_meta"] = meta
    config["additional"]["x_transformers_version"] = str(XT_VERSION)
    return meta


def amp_dtype(precision) -> "torch.dtype | None":
    if isinstance(precision, torch.dtype):
        return None if precision is torch.float32 else precision
    head = str(precision).split(",")[0].strip().lower()
    if head.startswith("bf16"):
        return torch.bfloat16
    if head.startswith("16"):
        return torch.float16
    return None


def _expand_geom_encoder(geom: dict | None, n_prefix: int) -> None:
    """Expands ``model_args.geom_encoder`` in place: layers repeated, ``hide_scalars`` on by default, ``n_prefix``."""
    if geom is None:
        return
    geom["layers"] = [[float(v) for v in row] for row in geom.get("layers") or []] * int(geom.pop("repeat", 1))
    geom.setdefault("hide_scalars", True)
    geom["n_prefix"] = n_prefix


def _edep_ae_spec(model_conf: dict) -> dict | None:
    """Loads the autoencoder that ``model_conf.edep_ae`` names into the config.

    ``{path: <scripts/train_edep_ae.py output>}`` is loaded once and replaced by its ``{"config", "state_dict"}`` (the
    path kept), so the saved flow config carries the autoencoder. Returns the spec, or None without the key.
    """
    spec = model_conf.get("edep_ae")
    if spec is not None and "state_dict" not in spec:
        spec = {**torch.load(spec["path"], map_location="cpu", weights_only=False), "path": spec["path"]}
        model_conf["edep_ae"] = spec
    return spec


# base_conf flags a base_head_lik config must set: tanh_theta is the angular distribution its likelihood assumes,
# the others name what that head always does. And the conditioning scalars its inputs are built from.
_LIK_FLAGS = ("tanh_theta", "base_head_nout", "ang_shape", "ang_nout", "ang_species")
_BASE_HEAD_SCALARS = ("Z", "A", "Size")


def _check_lego(config: dict, c: dict, cond_scalars: tuple, len_buckets, state_dict: dict | None) -> None:
    """Raises at the first rule a flow config breaks.

    Raises:
        KeyError:   fresh training without ``max_energy`` or ``cutoff_mev`` in meta.json or the config.
        ValueError: any other rule.
    """
    model_conf, dl_conf = config["model_conf"], config["dl_conf"]
    model_args = model_conf["model_args"]
    if state_dict is None:   # fresh training
        dpath = dl_conf["lds_args"]["data"]
        if model_conf["max_energy"] is None:
            raise KeyError(f"max_energy missing from {dpath}/meta.json and model_conf; "
                           "regenerate meta.json or set model_conf['max_energy'].")
        if dl_conf["lds_args"]["cutoff_mev"] is None:
            raise KeyError(f"cutoff_mev missing from {dpath}/meta.json and dl_conf.lds_args.")
    geom = model_args.get("geom_encoder")
    if geom is not None:
        layers = geom["layers"]
        if not layers or any(len(r) != 4 or min(r) <= 0 for r in layers):
            raise ValueError("model_args.geom_encoder.layers must be rows of positive [thickness mm, density, Z, A].")

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

    base_conf = config.get("base_conf") or {}
    if base_conf.get("base_head_lik", False):
        missing = [k for k in _LIK_FLAGS if not base_conf.get(k, False)]
        if missing or base_conf.get("scale_dist") != "sm_norm":
            raise ValueError(f"base_head_lik needs scale_dist sm_norm and {_LIK_FLAGS} (its two passes); got "
                             f"scale_dist {base_conf.get('scale_dist')!r}, missing {missing}.")
        if base_conf.get("base_head") is None and c["base_pretrain_batches"] <= 0:
            raise ValueError("base_head_lik needs base_pretrain_batches > 0 or a saved base_conf.base_head.")
        if not all(s in cond_scalars for s in _BASE_HEAD_SCALARS):
            raise ValueError(f"base_head needs the {_BASE_HEAD_SCALARS} conditioning scalars; "
                             f"got cond_scalars={cond_scalars}.")
        hidden = base_conf.get("base_head_hidden")
        if hidden is not None and len(hidden) != 2:
            raise ValueError(f"base_head_lik needs base_head_hidden [trunk, group], got {hidden}.")

    len_mode = dl_conf.get("len_mode", "split")
    len_split = dl_conf.get("len_split", 8)
    if len_buckets is not None:
        if len_mode not in ("split", "group"):
            raise ValueError(f"dl_conf.len_mode must be 'split' or 'group', got {len_mode!r}.")
        if len_mode == "split" and dl_conf["bs"] % len_split:
            raise ValueError(f"dl_conf.len_split={len_split} must divide dl_conf.bs={dl_conf['bs']}.")

    model_args = model_conf.get("model_args", {})
    if model_args.get("cond_pool") and (model_args.get("geom_encoder") is not None
                                        or not model_args.get("time_cond", True)
                                        or c["reflow_path"] or c["reflow_start_epoch"] > 0):
        raise ValueError("model_args.cond_pool does not go with geom_encoder (both stand in for the cond-scalar "
                         "tokens), nor with time_cond: false or reflow (their forward does not pool).")

    if model_conf.get("edep_ae") is not None:
        # len_mode group keeps a batch whole and in order (the per-layer targets stay with their events); split
        # reorders and runs pieces, which do not carry the latent token
        if ((len_buckets is not None and len_mode != "group") or c["reflow_path"] or c["reflow_start_epoch"] > 0
                or (model_conf.get("mask_conf") or {}).get("p_forward", 1.0) < 1):
            raise ValueError("model_conf.edep_ae does not support training len_buckets in split mode, reflow or "
                             "inverse-mask events (mask_conf.p_forward < 1) yet.")
        if state_dict and "lat_in.weight" not in state_dict:   # the field loads strict=False, which would hide it
            raise ValueError("model_conf.edep_ae checkpoint without the latent token (lat_in): the 16-token "
                             "prototype (feat/edep-latents) is not supported.")


def _check_mult(mm_conf: dict, dl_conf: dict, meta: dict) -> None:
    """Raises at the first rule a fresh multiplicity config breaks.

    Raises:
        ValueError: ``mm_conf.bs`` disagrees with ``dl_conf.bs``, or ``mm_conf.cond_scalars`` with the dataset's.
    """
    if "bs" in mm_conf and dl_conf.get("bs", mm_conf["bs"]) != mm_conf["bs"]:
        raise ValueError(f"mm_conf.bs={mm_conf['bs']} (the loader's) != dl_conf.bs={dl_conf['bs']}: set one")
    if meta.get("cond_scalars") and tuple(meta["cond_scalars"]) != tuple(mm_conf["cond_scalars"]):
        raise ValueError(f"mm_conf.cond_scalars={tuple(mm_conf['cond_scalars'])} != dataset layout "
                         f"{tuple(meta['cond_scalars'])}")
