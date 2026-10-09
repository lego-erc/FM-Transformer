"""Characterisation tests for the old-artefact support paths.

These fire only when loading checkpoints/configs from earlier vintages, which
no other test does (``test_generate.py`` is the only one that loads real
checkpoints and it needs network access). Written before the compat code was
moved to ``legofmt/compat.py``, so the move is verified rather than assumed.

``test_xt_compat.py`` already covers the qk-norm scale path; this file covers
the rest.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.mod_comps.config import (
    build_manifold, resolve_legoltng_config,
)

from test_modules_direct import _tiny_config


# --- config keys -----------------------------------------------------------

def test_ntokens_is_renamed_to_max_seq_l() -> None:
    """Pre-refactor checkpoints spelled max_seq_l 'ntokens'."""
    c = _tiny_config()
    ma = c["config"]["model_conf"]["model_args"]
    ma["ntokens"] = ma.pop("max_seq_l")
    rc = resolve_legoltng_config(c)
    assert rc.model_args["max_seq_l"] == 5
    assert "ntokens" not in rc.model_args


def test_project_in_checkpoint_sets_dim_in_out() -> None:
    """vf.project_in/out linears only rebuild if dim_in_out is set from h_dim."""
    c = _tiny_config()
    c["state_dict"] = {"vf.project_in.weight": torch.zeros(16, 7)}
    rc = resolve_legoltng_config(c)
    assert rc.model_args["dim_in_out"] == rc.model_args["h_dim"]


def test_no_project_in_leaves_dim_in_out_unset() -> None:
    rc = resolve_legoltng_config(_tiny_config())
    assert "dim_in_out" not in rc.model_args


# --- manifold spec ---------------------------------------------------------

def test_string_manifold_spec_still_builds_and_warns() -> None:
    with pytest.warns(DeprecationWarning, match="String manifold specs"):
        man = build_manifold("ProductManifold([Euclidean(), Sphere(), Sphere()], (1, 3, 3))")
    ref = build_manifold([
        {"name": "euclidean", "dim": 1},
        {"name": "sphere", "dim": 3},
        {"name": "sphere", "dim": 3},
    ])
    assert man.ambient_dims == ref.ambient_dims
    assert [type(m).__name__ for m in man.manifolds] == [type(m).__name__ for m in ref.manifolds]


# --- CFMTrafo_x parameter names -------------------------------------------

_RENAMES = {
    "l_mask_": "cond_w_mask", "b_mask_": "cond_bi_mask", "bo_mask_": "cond_bo_mask",
    "l_types_": "cond_w_types", "b_types_": "cond_bi_types", "bo_types_": "cond_bo_types",
    "l_pdgids_": "cond_w_pdgids", "b_pdgids_": "cond_bi_pdgids", "bo_pdgids_": "cond_bo_pdgids",
}


def _trafo() -> CFMTrafo_x:
    return CFMTrafo_x(h_dim=16, nlayers=1, nhead=2, in_dim=7, max_seq_l=5,
                      ntypes=4, nvtypes=2, npdgids=4, ff_mult=1)


def test_legacy_param_names_are_remapped_on_load() -> None:
    """Old checkpoints carry l_mask_/b_mask_/... instead of cond_w_mask/..."""
    ref = _trafo()
    sd = ref.state_dict()
    inv = {new: old for old, new in _RENAMES.items()}
    legacy = {inv.get(k, k): v for k, v in sd.items()}
    assert any(k in _RENAMES for k in legacy), "fixture did not produce legacy keys"

    fresh = _trafo()
    with pytest.warns(DeprecationWarning, match="Remapping legacy CFMTrafo_x"):
        fresh.load_state_dict(legacy)
    for k, v in sd.items():
        assert torch.equal(fresh.state_dict()[k], v), k


def test_current_param_names_load_without_warning() -> None:
    ref, fresh = _trafo(), _trafo()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fresh.load_state_dict(ref.state_dict())
