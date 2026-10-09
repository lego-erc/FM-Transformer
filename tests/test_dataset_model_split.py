"""A stored dataset must carry no model parameters.

``DataPrep.prep`` writes the incoming energy and ``E_dep`` in MeV and the
outgoing energy as a cutoff-only log ratio; ``DataPrep.norm_e`` puts the two
MeV channels on the model scale when ``LEGODataset`` loads the file. So one
dataset can feed models with different ``max_energy``, and a dataset can be
generated before any model exists.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F

CUTOFF = 10.0
E_IN, E_OUT, E_DEP = 200.0, 50.0, 60.0
MANIFOLD = [{"name": "euclidean", "dim": 1},
            {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}]


def _cfg(max_energy=None):
    cfg = {"cutoff_mev": CUTOFF, "manifold": MANIFOLD, "proj_ray": False,
           "cond_scalars": ["Density"], "energy_kin": True}
    if max_energy is not None:
        cfg["max_energy"] = max_energy
    return cfg


def _raw(n=4):
    def row(e, d, p, pid):
        return [e, *[e * c for c in d], *p, pid]
    inc = torch.tensor([[row(E_IN, (1, 0, 0), (-50, 0, 0), 2212.0)]]).repeat(n, 1, 1)
    out = torch.tensor([[row(E_OUT, (0, 1, 0), (0, 50, 0), 22.0)]]).repeat(n, 2, 1)
    return {"per_particle": {"Incoming": inc, "Outgoing": out},
            "per_event": {"E_dep": torch.full((n,), E_DEP),
                          "Density": torch.full((n,), 3.0)}}


@pytest.fixture
def dataset_dir(tmp_path):
    """A dataset generated with no knowledge of any model's max_energy."""
    ds = LEGODataset(data=_raw(), prep=DataPrep(_cfg()), cutoff_mev=CUTOFF).data
    torch.save((ds.f.full, ds.m.full, ds.am.full), tmp_path / "data_prepped.pt")
    (tmp_path / "meta.json").write_text(json.dumps({
        "ntokens": ds.f.full.shape[1], "particles": [22.0, 2212.0],
        "particles_in": [2212.0], "cutoff_mev": CUTOFF,
        "cond_scalars": ["Density"], "energy_kin": True, "max_energy": 300.0,
    }))
    return tmp_path


def test_prep_needs_no_max_energy(dataset_dir) -> None:
    f = torch.load(dataset_dir / "data_prepped.pt", weights_only=False)[0]
    assert float(_F(f).in_p[..., 0, 0][0]) == pytest.approx(E_IN)
    assert float(_F(f).edep[0]) == pytest.approx(E_DEP)


@pytest.mark.parametrize("max_energy", [300.0, 1000.0])
def test_one_file_feeds_models_with_different_max_energy(dataset_dir, max_energy) -> None:
    d = LEGODataset(data=str(dataset_dir / "data_prepped.pt"),
                    prep=DataPrep(_cfg(max_energy))).data
    log_range = math.log(max_energy / CUTOFF)
    assert float(_F(d.f.full).in_p[..., 0, 0][0]) == pytest.approx(
        math.log(E_IN / CUTOFF) / log_range, abs=1e-5)
    assert float(_F(d.f.full).edep[0]) == pytest.approx(E_DEP / max_energy, abs=1e-6)


def test_outgoing_column_is_max_energy_independent(dataset_dir) -> None:
    want = 1 - math.log(E_OUT / CUTOFF) / math.log(E_IN / CUTOFF)
    for max_energy in (300.0, 1000.0):
        d = LEGODataset(data=str(dataset_dir / "data_prepped.pt"),
                        prep=DataPrep(_cfg(max_energy))).data
        assert float(_F(d.f.full).out_p[..., 0, 0][0]) == pytest.approx(want, abs=1e-5)


def test_norm_e_does_not_mutate_the_caller(dataset_dir) -> None:
    stored = torch.load(dataset_dir / "data_prepped.pt", weights_only=False)
    before = stored[0].clone()
    DataPrep(_cfg(300.0)).norm_e(stored)
    assert torch.equal(stored[0], before)


@pytest.mark.parametrize("prep", [None, lambda x: x])
def test_loading_without_a_dataprep_gives_the_file_verbatim(dataset_dir, prep) -> None:
    # inspection / plotting callers legitimately want the stored tensors
    d = LEGODataset(data=str(dataset_dir / "data_prepped.pt"), prep=prep).data
    assert float(_F(d.f.full).in_p[..., 0, 0][0]) == pytest.approx(E_IN)   # MeV
    assert float(_F(d.f.full).edep[0]) == pytest.approx(E_DEP)


def test_setup_normalises_the_training_data(dataset_dir) -> None:
    """LEGOLtng.setup must pass a DataPrep -- forgetting it is how the stored
    file silently reached the model in MeV before."""
    from legofmt.main.modules import LEGOLtng
    cfg = {"dl_conf": {"lds_args": {"data": str(dataset_dir), "cutoff_mev": CUTOFF},
                       "bs": 2, "num_workers": 0},
           "val_conf": {"val_frac": 0.5, "seed": 0},
           "base_conf": {"kappa": 8.0, "base_dist": "poles", "scale_dist": "uniform"},
           "model_conf": {"manifold": MANIFOLD, "max_energy": 300.0,
                          "model_args": {"h_dim": 16, "nlayers": 1, "nhead": 2,
                                         "in_dim": 7, "ff_mult": 1, "dropout": 0.0}},
           "opt_conf": {"opt": "schedulefree", "lr": 1e-3}}
    model = LEGOLtng(cfg)
    model.setup()
    f = model._train_ds.dataset.data.f
    assert float(_F(f.full).in_p[..., 0, 0][0]) == pytest.approx(
        math.log(E_IN / CUTOFF) / math.log(300.0 / CUTOFF), abs=1e-5)
    assert float(_F(f.full).edep[0]) == pytest.approx(E_DEP / 300.0, abs=1e-6)



# --- legacy files ---------------------------------------------------------
# Datasets written before the split (everything up to rp_kin_*) already carry
# the normalisation. norm_e detects that and passes them through, so they load
# correctly instead of being normalised a second time (which collapses every
# incoming energy to 0 -- log of an already-normalised value is negative).

def _legacy(dataset_dir, max_energy=300.0):
    """The same events, stored the pre-split way: both channels normalised."""
    return DataPrep(_cfg(max_energy)).norm_e(
        torch.load(dataset_dir / "data_prepped.pt", weights_only=False))


def test_already_normalised_file_is_passed_through(dataset_dir) -> None:
    legacy = _legacy(dataset_dir)
    with pytest.warns(DeprecationWarning, match="already carries"):
        again = DataPrep(_cfg(300.0)).norm_e(legacy)
    assert torch.equal(again[0], legacy[0])


def test_mev_file_is_still_normalised(dataset_dir) -> None:
    stored = torch.load(dataset_dir / "data_prepped.pt", weights_only=False)
    assert float(_F(stored[0]).in_p[..., 0, 0][0]) == pytest.approx(E_IN)  # MeV
    out = DataPrep(_cfg(300.0)).norm_e(stored)
    assert float(_F(out[0]).in_p[..., 0, 0][0]) == pytest.approx(
        math.log(E_IN / CUTOFF) / math.log(300.0 / CUTOFF), abs=1e-5)


def test_legacy_file_loads_through_LEGODataset(dataset_dir, tmp_path) -> None:
    torch.save(_legacy(dataset_dir), tmp_path / "legacy.pt")
    with pytest.warns(DeprecationWarning):
        d = LEGODataset(data=str(tmp_path / "legacy.pt"), prep=DataPrep(_cfg(300.0))).data
    assert float(_F(d.f.full).in_p[..., 0, 0][0]) == pytest.approx(
        math.log(E_IN / CUTOFF) / math.log(300.0 / CUTOFF), abs=1e-5)
    assert float(_F(d.f.full).edep[0]) == pytest.approx(E_DEP / 300.0, abs=1e-6)
