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

PDGIDS = torch.tensor([-11, 11, 22], dtype=torch.int64)
FULL_SCALARS = ("Density", "Z", "A", "Size")


def _config(cond_scalars, *, base_dist_loss=0.0, base_pretrain_batches=300,
            base_head=None, scale_dist="sm_norm"):
    base_conf = {"kappa": 8.0, "bs_frac": 0.0, "base_dist": "poles",
                 "scale_dist": scale_dist, "tanh_theta": True, "sm_scale": 0.5}
    if base_head is not None:
        base_conf["base_head"] = base_head
        base_conf["base_head_frozen"] = True
    return {"state_dict": {}, "config": {
        "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
        "val_conf": {"val_frac": 0.01, "seed": 0},
        "base_conf": base_conf,
        "model_conf": {
            "manifold": [{"name": "euclidean", "dim": 1},
                         {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}],
            "max_energy": 300.0, "pdgids": PDGIDS, "cond_scalars": cond_scalars,
            "base_dist_loss": base_dist_loss,
            "base_pretrain_batches": base_pretrain_batches,
            "model_args": {"h_dim": 16, "nlayers": 1, "nhead": 2, "in_dim": 7,
                           "max_seq_l": 9, "ntypes": len(cond_scalars) + 3,
                           "nvtypes": 2, "npdgids": PDGIDS.numel() + 1,
                           "ff_mult": 1, "dropout": 0.0}},
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3}}}


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


def test_explicit_base_dist_loss_without_scalars_is_a_named_error() -> None:
    with pytest.raises(ValueError, match="conditioning scalars"):
        LEGOLtng(_config(("Density",), base_dist_loss=1.0))


def test_saved_head_without_scalars_is_a_named_error() -> None:
    with pytest.raises(ValueError, match="conditioning scalars"):
        LEGOLtng(_config(("Density",), base_pretrain_batches=0,
                         base_head=_saved_head()))


@pytest.mark.parametrize("bdl,bpb", [(1.0, 300), (0.0, 300), (1.0, 0)])
def test_built_when_requested_and_feedable(bdl, bpb) -> None:
    assert hasattr(LEGOLtng(_config(FULL_SCALARS, base_dist_loss=bdl,
                                    base_pretrain_batches=bpb)), "base_head")


def test_saved_head_is_rebuilt_when_both_knobs_are_off() -> None:
    # continuing from a pretrained+frozen head: dropping it would silently
    # revert the base distribution to the static base_conf sm_scale/kappa.
    hs = _saved_head()
    model = LEGOLtng(_config(FULL_SCALARS, base_dist_loss=0.0,
                             base_pretrain_batches=0, base_head=hs))
    assert hasattr(model, "base_head")
    assert torch.equal(model.base_head[-1].weight, hs["2.weight"])
    assert not model.base_head[-1].weight.requires_grad  # base_head_frozen


def test_no_head_without_sm_norm() -> None:
    assert not hasattr(
        LEGOLtng(_config(FULL_SCALARS, base_dist_loss=1.0, scale_dist="uniform")),
        "base_head",
    )
