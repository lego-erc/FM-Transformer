"""``EnergyProjections.norm_out`` / ``denorm_out``: the optional log scale for the
outgoing fractional energy loss, and its round trip through ``DataPrep.cc_trafo``."""

import pytest
import torch

from legofmt.data.prep import DataPrep
from legofmt.geometry.energy_proj import EnergyProjections

CFG = {
    "manifold": [{"name": "euclidean", "dim": 1}, {"name": "sphere", "dim": 3},
                 {"name": "sphere", "dim": 3}],
    "cutoff_mev": 10.0, "max_energy": 1000.0, "cond_scalars": ("Density",),
    "proj_ray": False,
}
LO = 1e-7
# Smallest real losses measured on rp_had_060926 at E_in = 1 GeV: charged
# particles bottom out at u ~ 2.2e-4 (Cu) / 1.2e-3 (Ar); the neutron pass-through
# tail reaches u ~ 8e-7. 0 is the genuine no-interaction atom.
U = torch.tensor([0.0, 1e-8, 8e-7, 2.2e-4, 1.2e-3, 0.5, 1.0])


def _pen(lo):
    return EnergyProjections(cutoff_mev=10.0, max_energy=1000.0, e_log_min=lo)


def test_default_is_the_raw_linear_scale():
    assert torch.equal(_pen(None).norm_out(U), U)
    assert torch.equal(_pen(None).denorm_out(U), U)


def test_log_scale_lifts_the_small_losses_off_zero():
    out = _pen(LO).norm_out(U)
    assert out[0] == 0.0                      # no interaction stays the atom
    assert out[1] == 0.0                      # below the floor -> atom
    assert out[2] > 0.1                       # neutron tail, clear of the atom
    assert out[3] > 0.45                      # charged onset, ~half the axis up
    assert torch.all(out[2:].diff() > 0)      # strictly monotone above the floor
    assert out[-1] == pytest.approx(1.0)      # u = 1 (fully stopped) stays 1


@pytest.mark.parametrize("lo", [None, LO])
def test_round_trip(lo):
    pen = _pen(lo)
    rt = pen.denorm_out(pen.norm_out(U))
    keep = U >= LO
    assert torch.allclose(rt[keep], U[keep], rtol=1e-5)
    assert rt[0] == 0.0


def test_floor_must_sit_below_the_neutron_tail():
    """1e-5 was the tempting value; it silently deletes the neutron small-loss
    population, which is the regression ``edep_log_min`` exists to avoid."""
    assert _pen(1e-5).norm_out(U)[2] == 0.0    # u = 8e-7 destroyed
    assert _pen(LO).norm_out(U)[2] > 0.0       # preserved


def test_to_mev_inverts_cc_trafo_end_to_end():
    """A particle at E_out through cc_trafo and back out of to_mev is itself."""
    e_in = 1000.0
    e_out = torch.tensor([999.9, 998.4, 983.3, 500.0, 10.0])
    # cc_trafo indexes dim 1 as the slot axis: row 0 is the incoming particle,
    # rows 1: are the outgoing ones it is normalised against.
    mev = torch.cat((torch.tensor([e_in]), e_out))[None, :, None]
    cc = torch.zeros(1, mev.shape[1], 6)
    cc[..., 0] = 1.0                                     # unit momentum along x
    cc[..., 3] = 1.0                                     # unit position
    for lo in (None, LO):
        prep = DataPrep({**CFG, "e_log_min": lo})
        v = prep.cc_trafo(cc.clone(), e_kin=mev)[0, 1:, 0]
        e_in_sc = prep.pen.to_scalar_e(torch.tensor([e_in]))
        assert torch.allclose(prep.pen.to_mev(v, e_in_sc), e_out, rtol=1e-4)


def test_log_scale_separates_copper_from_argon():
    """The point of the change: at E_in = 1 GeV a copper loss (1.6 MeV) and an
    argon loss (16.7 MeV) sit 3.3e-3 apart in u, under the channel's ~1e-3 noise.
    The log scale pushes them apart by two orders of magnitude."""
    u = torch.tensor([3.47e-4, 3.65e-3])                 # cu, ar
    assert (u[1] - u[0]) < 4e-3
    v = _pen(LO).norm_out(u)
    assert (v[1] - v[0]) > 0.1
