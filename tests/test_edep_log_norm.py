"""``DataPrep.norm_edep`` / ``denorm_edep``: the optional log scale for E_dep."""

import pytest
import torch

from legofmt.data.prep import DataPrep

CFG = {
    "manifold": [{"name": "euclidean", "dim": 1}, {"name": "sphere", "dim": 3},
                 {"name": "sphere", "dim": 3}],
    "cutoff_mev": 10.0, "max_energy": 1000.0, "cond_scalars": ("Density",),
}
MEV = torch.tensor([0.0, 1e-6, 2.9e-4, 1.574, 10.0, 81.0, 1000.0, 5000.0])


def _prep(lo):
    return DataPrep({**CFG, "edep_log_min": lo})


def test_default_is_the_linear_scale():
    assert torch.allclose(_prep(None).norm_edep(MEV), MEV / 1000.0)


def test_log_scale_lifts_the_low_end_off_zero():
    out = _prep(1e-6).norm_edep(MEV)
    assert out[0] == 0.0                       # zero deposit stays the atom
    assert out[1] == 0.0                       # at the floor -> atom
    assert out[2] > 0.25                       # smallest real deposit, clear of 0
    assert torch.all(out[2:].diff() >= 0)      # monotone above the floor
    assert out.max() <= 1.0                    # bounded, so > max_energy clamps


@pytest.mark.parametrize("lo", [None, 1e-6])
def test_round_trip(lo):
    prep = _prep(lo)
    rt = prep.denorm_edep(prep.norm_edep(MEV))
    keep = (MEV > 1e-5) & (MEV <= 1000.0)
    assert torch.allclose(rt[keep], MEV[keep], rtol=1e-5)
    assert rt[0] == 0.0


def test_zero_is_never_created_or_destroyed_above_the_floor():
    prep = _prep(1e-6)
    e = torch.tensor([0.0, 1e-3, 0.0, 50.0])
    assert (prep.norm_edep(e) == 0).tolist() == [True, False, True, False]
