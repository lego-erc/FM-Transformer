"""Everything that exists only so older artefacts keep working.

Collected here so the current implementation reads clean and so the support
surface is visible in one place: when a vintage is dropped, delete its entry
here and the one call site that references it.

Three kinds, and they behave differently:

1. **state_dict migrations** -- one-shot, at load time. Pure functions from an
   old-shaped checkpoint to the current layout.
2. **runtime switches** -- not migrations. These change how an old checkpoint
   *runs*, on every forward, for as long as it is loaded.
3. **format deprecations** -- older spellings of configs and datasets.

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

from legofmt.data.struct import _F
from legofmt.geometry.product_manifold import ProductManifold


# ---------------------------------------------------------------------------
# 1. state_dict migrations
# ---------------------------------------------------------------------------

def rename_ntokens(model_args: dict) -> None:
    """Pre-refactor checkpoints spelled ``max_seq_l`` as ``ntokens``."""
    if "ntokens" in model_args:
        model_args["max_seq_l"] = model_args.pop("ntokens")


def apply_legacy_projection_in_out(model_args: dict, state_dict: dict) -> None:
    """Legacy checkpoints carry ``vf.project_in/out`` linears; rebuild them by
    setting ``dim_in_out`` so ``load_state_dict`` finds matching layers."""
    if any(k.startswith("vf.project_in.") for k in state_dict):
        model_args["dim_in_out"] = model_args["h_dim"]


def migrate_legacy_mult_heads(
    state_dict: dict, max_seq_len: int, max_particles: int
) -> dict:
    """Pre-fusion checkpoints stored per-position ModuleLists (``embd_in_.{i}``,
    ``proj_out_.{i}``); remap them onto the fused single-table layout."""
    has_legacy_proj = "proj_out_.0.weight" in state_dict
    has_legacy_embd = "embd_in_.0.weight" in state_dict
    if not (has_legacy_proj or has_legacy_embd):
        return state_dict

    out = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("proj_out_.") or k.startswith("embd_in_."))
    }

    if has_legacy_proj:
        weights = [state_dict[f"proj_out_.{i}.weight"] for i in range(max_seq_len)]
        biases = [state_dict[f"proj_out_.{i}.bias"] for i in range(max_seq_len)]
        out["proj_out_w"] = torch.stack([w.t().contiguous() for w in weights], dim=0)
        out["proj_out_b"] = torch.stack(biases, dim=0)

    if has_legacy_embd:
        embd_weights = [
            state_dict[f"embd_in_.{i}.weight"] for i in range(max_seq_len - 1)
        ]
        out["embd_in_.weight"] = torch.cat(embd_weights, dim=0)

    return out


# Pre-refactor CFMTrafo_x parameter names -> current names.
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
    """``load_state_dict`` pre-hook remapping the pre-refactor parameter names."""
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


def load_legacy_base_head(head: nn.Sequential, hs: dict) -> None:
    """A ``base_head`` saved with fewer outputs than the current one: copy the
    rows that exist and leave the rest at their init values."""
    n_old = hs["2.weight"].shape[0]
    head[0].load_state_dict({"weight": hs["0.weight"], "bias": hs["0.bias"]})
    head[-1].weight[:n_old].copy_(hs["2.weight"])
    head[-1].bias[:n_old].copy_(hs["2.bias"])


# ---------------------------------------------------------------------------
# 2. runtime switches -- these keep acting for as long as the model is loaded
# ---------------------------------------------------------------------------

# x-transformers < 2.25.5 applied ``attn_qk_norm_scale`` (default 10) to q, to k
# *and* as the kernel scale, so attention logits were scale**3 * cos(theta);
# 2.25.5+ applies it once. Checkpoints record the library they were trained
# under so old ones keep their effective scale when loaded by a newer library.
XT_VERSION = Version(_dist_version("x-transformers"))
XT_QK_NORM_FIX = Version("2.25.5")


def qk_norm_scale_compat(model_args: dict, additional: dict) -> None:
    saved = additional.get("x_transformers_version")
    old_ckpt = saved is None or Version(saved) < XT_QK_NORM_FIX
    if model_args.get("attn_qk_norm") and old_ckpt and XT_VERSION >= XT_QK_NORM_FIX:
        model_args["attn_qk_norm_scale"] = model_args.get("attn_qk_norm_scale", 10) ** 3
    elif model_args.get("attn_qk_norm") and not old_ckpt and XT_VERSION < XT_QK_NORM_FIX:
        raise RuntimeError(
            f"checkpoint trained with x-transformers {saved} (single qk_norm scale) "
            f"cannot be loaded by {XT_VERSION}; use x-transformers >= {XT_QK_NORM_FIX}."
        )


# bf16 rounding of unit q, k perturbs cos(theta) by ~2**-8, i.e. the logits by
# ~qk_norm_scale/256: negligible at the library default 10, but O(1) for the
# pre-2.25.5 checkpoints whose effective scale is 1000 (0.27 -> 0.03 rel. error
# on kin_020926 with fp32 attention). Only those get the fp32 kernel.
FP32_ATTN_QK_SCALE = 100


def needs_fp32_attention(model_args: dict) -> bool:
    return bool(model_args.get("attn_qk_norm")) and model_args.get("attn_qk_norm_scale", 10) > FP32_ATTN_QK_SCALE


def fp32_attention(module: nn.Module) -> None:
    """Run every ``Attend`` in fp32 even under autocast (see ``needs_fp32_attention``)."""
    def _wrap(fwd):
        def forward(q, k, v, *args, **kwargs):
            with torch.autocast(q.device.type, enabled=False):
                return fwd(q.float(), k.float(), v.float(), *args, **kwargs)
        return forward

    for m in module.modules():
        if isinstance(m, Attend):
            m.forward = _wrap(m.forward)


# ---------------------------------------------------------------------------
# 3. format deprecations
# ---------------------------------------------------------------------------

# Restricted eval namespace for the legacy string spec form.
MANIFOLD_EVAL_NS: dict[str, Any] = {
    "ProductManifold": ProductManifold,
    "Euclidean": Euclidean,
    "Sphere": Sphere,
}


def build_manifold_from_string(spec: str) -> ProductManifold:
    warnings.warn(
        "String manifold specs are deprecated; use a list of factor dicts "
        "([{'name': 'euclidean', 'dim': 3}, ...]).",
        DeprecationWarning,
        stacklevel=3,
    )
    return eval(spec, {"__builtins__": {}}, MANIFOLD_EVAL_NS)


def dataset_already_normalised(f: torch.Tensor, max_energy: float) -> bool:
    """Files written before the split (everything up to ``rp_kin_*``) store both
    channels already normalised and must not be normalised again.

    A normalised file has every incoming energy in ``[0, 1]``; a MeV file reaches
    ``max_energy``, so the two only collide when the whole incoming spectrum sits
    below 1 MeV -- which ``max_energy > 1`` rules out.
    """
    if _F(f).in_cc[..., 0].max() <= 1.0 < max_energy:
        warnings.warn(
            "dataset already carries the energy normalisation; skipping norm_e. "
            "Regenerate it to store MeV and decouple it from max_energy.",
            DeprecationWarning, stacklevel=4,
        )
        return True
    return False
