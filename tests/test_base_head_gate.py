"""The base_head construction gate.

``base_head_params`` reads the Z/A/Size conditioning scalars, so a head is only
buildable when the dataset carries them. The gate must not crash a model that
never asked for one, must fail loudly for a model that did, and must rebuild a
head that a previous run pretrained and saved.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.main.modules import LEGOLtng
from test_helpers import _tiny_config

PDGIDS = torch.tensor([-11, 11, 22], dtype=torch.int64)
FULL_SCALARS = ("Density", "Z", "A", "Size")


def _config(cond_scalars=FULL_SCALARS, base_pretrain_batches: int = 300, **base_conf) -> dict:
    cfg = _tiny_config({"nlayers": 1, "max_seq_l": 9, "ntypes": len(cond_scalars) + 3, "npdgids": PDGIDS.numel() + 1},
                       pdgids=PDGIDS, cond_scalars=cond_scalars, base_pretrain_batches=base_pretrain_batches)
    cfg["config"]["base_conf"].update({"scale_dist": "sm_norm", "sm_scale": 0.5, **base_conf})
    return cfg


def _saved_head() -> dict:
    return torch.nn.Sequential(
        torch.nn.Linear(4 + PDGIDS.numel(), 16), torch.nn.Mish(),
        torch.nn.Linear(16, 4),
    ).state_dict()


def test_default_pretrain_does_not_force_the_scalars() -> None:
    # base_pretrain_batches defaults to 300, i.e. nobody asked for a head; a
    # ("Density",) dataset must still build and generate.
    model = LEGOLtng(_config(("Density",)))
    assert not hasattr(model, "base_head")


def test_saved_head_without_scalars_is_a_named_error() -> None:
    with pytest.raises(ValueError, match="conditioning scalars"):
        LEGOLtng(_config(("Density",), base_pretrain_batches=0, base_head=_saved_head()))


def test_built_when_requested_and_feedable() -> None:
    assert hasattr(LEGOLtng(_config()), "base_head")


def test_no_head_when_nothing_asks() -> None:
    # base_dist_loss used to build one on its own; with co-training gone the only
    # asks left are base_pretrain_batches and a saved head.
    assert not hasattr(LEGOLtng(_config(base_pretrain_batches=0)), "base_head")


def test_saved_head_is_rebuilt_and_always_frozen() -> None:
    # continuing from a pretrained head: dropping it would silently revert the base
    # distribution to the static base_conf sm_scale/kappa. It comes back frozen even
    # when a pre-2026-09 config says base_head_frozen: False, because the flow never
    # trains the head -- an unfrozen one would sit in the optimizer with no gradient.
    hs = _saved_head()
    model = LEGOLtng(_config(base_pretrain_batches=0, base_head=hs, base_head_frozen=False))
    assert hasattr(model, "base_head")
    assert torch.equal(model.base_head[-1].weight, hs["2.weight"])
    assert not model.base_head[-1].weight.requires_grad


def test_base_head_hidden_sets_the_layer_widths() -> None:
    # the 16-unit default under-fits E_dep's prior (sig 7.8x the Geant4 sd on the
    # 1 GeV proton survivors), so the hidden layers are configurable
    head = LEGOLtng(_config(base_head_hidden=[32, 8])).base_head
    assert [m.out_features for m in head if isinstance(m, torch.nn.Linear)] == [32, 8, 4]
    assert not head[-1].weight.any()   # zero-init output layer, priors in the bias


def test_saved_deep_head_is_reloaded_frozen() -> None:
    hs = {k: torch.randn_like(v) for k, v in LEGOLtng(_config(base_head_hidden=[32, 8])).base_head.state_dict().items()}
    head = LEGOLtng(_config(base_pretrain_batches=0, base_head=hs, base_head_hidden=[32, 8])).base_head
    assert torch.equal(head[-1].weight, hs["4.weight"])
    assert not head[-1].weight.requires_grad


def test_saved_head_of_other_hidden_widths_is_rebuilt_fresh() -> None:
    # same depth and first layer, another inner width: the keys match, but the head must not be loaded
    hs = {k: torch.randn_like(v) for k, v in LEGOLtng(_config(base_head_hidden=[32, 8])).base_head.state_dict().items()}
    head = LEGOLtng(_config(base_pretrain_batches=0, base_head=hs, base_head_hidden=[32, 16])).base_head
    assert [m.out_features for m in head if isinstance(m, torch.nn.Linear)] == [32, 16, 4]
    assert not head[-1].weight.any()


def test_no_head_without_sm_norm() -> None:
    assert not hasattr(LEGOLtng(_config(scale_dist="uniform")), "base_head")
