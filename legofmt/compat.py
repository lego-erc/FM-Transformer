"""Adapters for older checkpoints, configs and dataset files, and the patches to x-transformers.

Kept in one file so that the support for older artefacts is visible in one place: when one is dropped, delete its
entry here and its call sites. The model code of options that left the recipe is in ``legofmt/legacy/``.

    1. state_dict migrations   applied once, at load: an older checkpoint's keys and shapes -> the current layout
    2. runtime switches        change how a loaded model runs, on every forward, for as long as it is loaded
    3. format deprecations     older spellings of config keys, and dataset files in an older format

Covered by ``tests/test_compat.py`` and ``tests/test_xt_compat.py``.
"""

from __future__ import annotations

import warnings
from importlib.metadata import version as _dist_version
from typing import Any

import torch
from torch import nn
from packaging.version import Version

from flow_matching.utils.manifolds import Euclidean, Sphere
from x_transformers.attend import Attend

from legofmt.data.struct import _F, set_layout
from legofmt.geometry.product_manifold import ProductManifold


# ---------------------------------------------------------------------------
# 1. state_dict migrations
# ---------------------------------------------------------------------------

def rename_ntokens(model_args: dict) -> None:
    """Renames ``model_args.ntokens``, as older checkpoints spell it, to ``max_seq_l``; in place."""
    if "ntokens" in model_args:
        model_args["max_seq_l"] = model_args.pop("ntokens")


def apply_legacy_projection_in_out(model_args: dict, state_dict: dict) -> None:
    """Sets ``model_args.dim_in_out = h_dim`` for a checkpoint that holds ``vf.project_in`` linears; in place.

    With ``dim_in_out`` set, the field is built with ``project_in`` / ``project_out`` (``legacy/cfm_trafo_x.py``), so
    ``load_state_dict`` finds them.
    """
    if any(k.startswith("vf.project_in.") for k in state_dict):
        model_args["dim_in_out"] = model_args["h_dim"]


def migrate_legacy_mult_heads(state_dict: dict, max_seq_len: int) -> dict:
    """Returns a multiplicity checkpoint's state_dict with its per-position layers merged into the current tables.

    An older checkpoint holds ``max_seq_len`` linears ``proj_out_.{i}`` and ``max_seq_len - 1`` embeddings
    ``embd_in_.{i}``, one per sequence position ``i``. They become ``proj_out_w`` (max_seq_len, h_dim, max_particles),
    ``proj_out_b`` (max_seq_len, max_particles) and the one table ``embd_in_.weight``, the per-position embeddings
    concatenated. A current state_dict is returned as it is.
    """
    has_legacy_proj = "proj_out_.0.weight" in state_dict
    has_legacy_embd = "embd_in_.0.weight" in state_dict
    if not (has_legacy_proj or has_legacy_embd):
        return state_dict

    out = {k: v for k, v in state_dict.items() if not k.startswith(("proj_out_.", "embd_in_."))}
    if has_legacy_proj:
        rows = range(max_seq_len)
        out["proj_out_w"] = torch.stack([state_dict[f"proj_out_.{i}.weight"].mT.contiguous() for i in rows])
        out["proj_out_b"] = torch.stack([state_dict[f"proj_out_.{i}.bias"] for i in rows])
    if has_legacy_embd:
        out["embd_in_.weight"] = torch.cat([state_dict[f"embd_in_.{i}.weight"] for i in range(max_seq_len - 1)])
    return out


# ``CFMTrafo_x`` parameter names in older checkpoints -> current names.
LEGACY_PARAM_RENAME: dict[str, str] = {
    "l_mask_":    "cond_w_mask",
    "b_mask_":    "cond_bi_mask",
    "bo_mask_":   "cond_bo_mask",
    "l_types_":   "cond_w_types",
    "b_types_":   "cond_bi_types",
    "bo_types_":  "cond_bo_types",
    "l_pdgids_":  "cond_w_pdgids",
    "b_pdgids_":  "cond_bi_pdgids",
    "bo_pdgids_": "cond_bo_pdgids",
}


def legacy_param_rename(state_dict, prefix, *_) -> None:
    """Renames an older checkpoint's ``CFMTrafo_x`` parameters in ``state_dict`` (``LEGACY_PARAM_RENAME``); in place.

    A ``load_state_dict`` pre-hook: ``prefix`` is the module's key prefix. Warns when it renames a key.
    """
    renames = {
        k: prefix + LEGACY_PARAM_RENAME[suf]
        for k in list(state_dict)
        if k.startswith(prefix) and (suf := k[len(prefix):]) in LEGACY_PARAM_RENAME
    }
    if not renames:
        return
    warnings.warn(
        "Remapping legacy CFMTrafo_x parameter keys "
        "(e.g. 'l_mask_' -> 'cond_w_mask'); re-save to silence.",
        DeprecationWarning, stacklevel=4,
    )
    for old, new in renames.items():
        state_dict[new] = state_dict.pop(old)


# ---------------------------------------------------------------------------
# 2. runtime switches: they act for as long as the model is loaded
# ---------------------------------------------------------------------------

# x-transformers < 2.25.5 applied ``attn_qk_norm_scale`` (default 10) to q, to k
# *and* as the kernel scale, so attention logits were scale**3 * cos(theta);
# 2.25.5+ applies it once. Checkpoints record the library they were trained
# under so old ones keep their effective scale when loaded by a newer library.
XT_VERSION = Version(_dist_version("x-transformers"))
XT_QK_NORM_FIX = Version("2.25.5")


def qk_norm_scale_compat(model_args: dict, additional: dict) -> None:
    """Cubes ``model_args.attn_qk_norm_scale`` for a checkpoint trained before x-transformers 2.25.5; in place.

    Only with ``attn_qk_norm`` and an installed x-transformers of 2.25.5 or later.

    Args:
        model_args: the transformer's arguments (the field's, or the multiplicity model's).
        additional: the config's ``additional`` section; its ``x_transformers_version`` is the version the checkpoint
            was trained with (absent: before 2.25.5).

    Raises:
        RuntimeError: a checkpoint trained with 2.25.5 or later meets an older installed x-transformers.
    """
    saved = additional.get("x_transformers_version")
    old_ckpt = saved is None or Version(saved) < XT_QK_NORM_FIX
    if model_args.get("attn_qk_norm") and old_ckpt and XT_VERSION >= XT_QK_NORM_FIX:
        model_args["attn_qk_norm_scale"] = model_args.get("attn_qk_norm_scale", 10) ** 3
    elif model_args.get("attn_qk_norm") and not old_ckpt and XT_VERSION < XT_QK_NORM_FIX:
        raise RuntimeError(
            f"checkpoint trained with x-transformers {saved} (single qk_norm scale) "
            f"cannot be loaded by {XT_VERSION}; use x-transformers >= {XT_QK_NORM_FIX}."
        )


# In bf16, rounding the unit vectors q and k shifts cos(theta) by ~2**-8, so the logits by ~qk_norm_scale / 256:
# negligible at the library default 10, but O(1) at the effective scale 1000 of a pre-2.25.5 checkpoint (relative
# error 0.27 in bf16, 0.03 with fp32 attention). Only those checkpoints get the fp32 kernel.
FP32_ATTN_QK_SCALE = 100


def needs_fp32_attention(model_args: dict) -> bool:
    """Returns whether a model needs fp32 attention: ``attn_qk_norm`` with a scale above ``FP32_ATTN_QK_SCALE``."""
    return bool(model_args.get("attn_qk_norm")) and model_args.get("attn_qk_norm_scale", 10) > FP32_ATTN_QK_SCALE


def fp32_attention(module: nn.Module) -> None:
    """Makes every ``Attend`` of ``module`` run in fp32 even under autocast (see ``needs_fp32_attention``)."""
    def _wrap(fwd):
        def forward(q, k, v, *args, **kwargs):
            with torch.autocast(q.device.type, enabled=False):
                return fwd(q.float(), k.float(), v.float(), *args, **kwargs)
        return forward

    for m in module.modules():
        if isinstance(m, Attend):
            m.forward = _wrap(m.forward)


def dynamic_shape_attention() -> None:
    """Patches x-transformers so that ``torch.compile(dynamic=True)`` gives one graph for every batch size and length.

    x-transformers masks ``Attention``'s output with ``einx.where('b n, b n d, -> b n d', mask, out, 0.)``, and einx
    traces per concrete shape, so dynamo specialises both dims and compiles one graph per bucket shape. The patch
    replaces that one pattern by the equivalent ``torch.where``. It acts on the whole process; a second call does
    nothing.
    """
    import types
    import x_transformers.x_transformers as xt

    einx = xt.einx
    if getattr(einx, "_legofmt_where", False):
        return

    def where(pattern, mask, out, val):
        if pattern != "b n, b n d, -> b n d":
            return einx.where(pattern, mask, out, val)
        return torch.where(mask.unsqueeze(-1), out, val)

    ns = {n: getattr(einx, n) for n in dir(einx) if not n.startswith("_")}
    xt.einx = types.SimpleNamespace(**ns | {"where": where, "_legofmt_where": True})


# ---------------------------------------------------------------------------
# 3. format deprecations
# ---------------------------------------------------------------------------

# The names a legacy string manifold spec may use: the namespace it is evaluated in.
MANIFOLD_EVAL_NS: dict[str, Any] = {
    "ProductManifold": ProductManifold,
    "Euclidean": Euclidean,
    "Sphere": Sphere,
}


def build_manifold_from_string(spec: str) -> ProductManifold:
    """Evaluates a legacy string manifold spec to its ``ProductManifold``, with a DeprecationWarning."""
    warnings.warn(
        "String manifold specs are deprecated; use a list of factor dicts "
        "([{'name': 'euclidean', 'dim': 3}, ...]).",
        DeprecationWarning,
        stacklevel=3,
    )
    return eval(spec, {"__builtins__": {}}, MANIFOLD_EVAL_NS)


def dataset_already_normalised(f: torch.Tensor, max_energy: float) -> bool:
    """Returns whether a dataset file already stores its incoming energy and E_dep on the model scale.

    Older files (every dataset up to ``rp_kin_*``) were normalised when they were written, and must not be normalised
    again. Such a file has every incoming energy in [0, 1]; a file in MeV is taken to have incoming energies above
    1 MeV. With ``max_energy <= 1`` the two cannot be told apart, and the file is taken to be in MeV. A file found
    normalised gives a DeprecationWarning.

    Args:
        f:          the dataset's tokens, as ``DataPrep.norm_e`` gets them.
        max_energy: the model's energy ceiling [MeV].
    """
    if _F(f).in_cc[..., 0].max() <= 1.0 < max_energy:
        warnings.warn(
            "dataset already carries the energy normalisation; skipping norm_e. "
            "Regenerate it to store MeV and decouple it from max_energy.",
            DeprecationWarning, stacklevel=4,
        )
        return True
    return False


# Older configs name the learned loss weights ``uncert_*`` and hold one ``lv`` per (t-bin, loss cell). The t-bins are
# gone (16 bins against 1 made no difference to the generated E_dep): a saved ``lv`` of ``bins * n_cells`` entries is
# reduced to one per loss cell by the mean of its variances ``exp(lv)``, because a converged Kendall cell sits at
# ``lv = log(L)``.
_RENAMED_KEYS = {
    "uncert_weighting": "learned_loss_weights",
    "uncert_min": "max_loss_weight",
    "uncert_min_flow": "max_loss_weight_flow",
    "uncert_lv": "loss_weights",
}


def migrate_loss_weight_keys(model_conf: dict) -> dict:
    """Renames an older ``model_conf``'s ``uncert_*`` keys and reduces a saved ``lv`` with t-bins to one per loss cell.

    Changes ``model_conf`` in place and returns it.
    """
    for old, new in _RENAMED_KEYS.items():
        if old in model_conf:
            model_conf.setdefault(new, model_conf.pop(old))
    model_conf.pop("uncert_bins", None)
    lv = (model_conf.get("loss_weights") or {}).get("lv")
    n_cells = 3 + int(model_conf.get("edep_cell", False)) + int(model_conf.get("edep_ae") is not None)  # + lat
    if lv is not None and lv.numel() > n_cells:
        model_conf["loss_weights"]["lv"] = torch.log(torch.exp(lv.reshape(-1, n_cells)).mean(0))
    return model_conf


def edep_margin_keys(conf: dict) -> tuple:
    """Returns ``conf``'s ``(edep_margin, edep_margin_species)``, also read under their older names ``edep_wall*``.

    Unset: ``(False, None)``.
    """
    return (conf.get("edep_margin", conf.get("edep_wall", False)),
            conf.get("edep_margin_species", conf.get("edep_wall_species")))


def edep_excess_from_data(config: dict) -> None:
    """Fills ``model_conf.edep_excess_mev`` from the dataset's tokens, in place.

    For a margin model on a dataset whose meta.json lacks ``edep_excess_max_mev``: the padded file
    ``dl_conf.data_path`` is read. A config without the margin token, or with ``edep_excess_mev`` set, is left as it is.
    """
    model_conf = config["model_conf"]
    if edep_margin_keys(model_conf)[0] and "edep_excess_mev" not in model_conf:
        from legofmt.data.write.norm import DataPrep
        set_layout(model_conf["cond_scalars"])
        f = torch.load(config["dl_conf"]["data_path"], map_location="cpu", mmap=True)[0]
        model_conf["edep_excess_mev"] = DataPrep.edep_excess_max(f)
