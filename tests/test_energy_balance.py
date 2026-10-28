"""odeint_conf.energy_balance: GenerateOut closes E_in = E_dep + sum E_out per event, keeping the generated E_dep.

A chained thin-plate model cannot resolve a particle that crosses a plate almost unchanged: the outgoing energy
of a 1 GeV proton losing 3 MeV sits 4e-4 from the channel's clamp, so the flow hands out 0 or ~28 MeV losses and
every e- call violates the balance by ~2.6%. On the 0.1 MeV-cutoff plate data Geant4's own balance is exact
(median residual 0), so the projection is safe there; it is opt-in because a 10 MeV cutoff loses ~7% below it.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F, set_layout
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.main.generate import _balance_energy

CUT, MAX, EDEP_MIN = 0.1, 2000.0, 1e-6
COND = ("Density", "Z", "A", "Size")


@pytest.fixture
def codec():
    dp = DataPrep({"cutoff_mev": CUT, "max_energy": MAX, "edep_log_min": EDEP_MIN, "cond_scalars": COND})
    return EnergyProjections(cutoff_mev=CUT, max_energy=MAX), dp


def _event(pen, dp, e_in: float, edep: float, out: list[tuple[int, float]], L: int = 9) -> torch.Tensor:
    """One (1, L, 8) model-space event: E_dep row, incoming row, outgoing rows (pdgid, MeV), NaN padding."""
    set_layout(COND)
    f = torch.full((1, L, 8), float("nan"))
    f[..., 7] = 0.0
    s = _F(f)
    s.non_p[...] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    s.edep[...] = dp.norm_edep(torch.tensor([edep]))
    e_in_n = pen.to_scalar_e(torch.tensor([[e_in]]))
    for i, (pid, e) in enumerate(out):
        s.out_p[0, i, 0] = 1 - pen.to_scalar_e(torch.tensor(e)) / e_in_n[0, 0]
        s.out_p[0, i, 1:7] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        s.out_p[0, i, 7] = pid
    return f


def _decode(pen, dp, f: torch.Tensor, e_in: float) -> tuple[float, torch.Tensor]:
    s = _F(f)
    live = s.out_p[0, :, 7] != 0
    e = pen.to_mev(s.out_p[0, live, 0], pen.to_scalar_e(torch.tensor(e_in)))
    return float(dp.denorm_edep(s.edep)[0]), e


def _run(pen, dp, f, e_in, pid):
    _balance_energy(f, torch.tensor([e_in]), torch.tensor([pid]), pen, dp)


def test_em_event_is_scaled_to_close_the_balance(codec) -> None:
    """e-: every outgoing energy shares the correction, E_dep stays the flow's."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 4.0, [(11, 700.0), (22, 300.0)])
    _run(pen, dp, f, 1000.0, 11)
    edep, e = _decode(pen, dp, f, 1000.0)
    assert edep == pytest.approx(4.0, rel=1e-4)
    assert float(e.sum()) + edep == pytest.approx(1000.0, rel=1e-5)
    assert float(e[0] / e[1]) == pytest.approx(700 / 300, rel=1e-4)


def test_pair_production_pays_the_pair_mass(codec) -> None:
    """Energies are kinetic: a conversion leaves E_in - E_dep - 2 m_e to its e+ e- (Geant4's residual is +1.022)."""
    pen, dp = codec
    f = _event(pen, dp, 100.0, 1.0, [(11, 50.0), (-11, 50.0)])
    _run(pen, dp, f, 100.0, 22)
    _, e = _decode(pen, dp, f, 100.0)
    assert float(e.sum()) == pytest.approx(100.0 - 1.0 - 1.022, rel=1e-5)


def test_positron_annihilation_releases_the_pair_mass(codec) -> None:
    """A positron that stops hands its 2 m_e to the photons (Geant4's residual is -1.022)."""
    pen, dp = codec
    f = _event(pen, dp, 5.0, 3.0, [(22, 1.5), (22, 1.5)])
    _run(pen, dp, f, 5.0, -11)
    _, e = _decode(pen, dp, f, 5.0)
    assert float(e.sum()) == pytest.approx(5.0 - 3.0 + 1.022, rel=1e-5)


@pytest.mark.parametrize("pid, expect", [(22, 0.5), (11, 0.5), (-11, 0.5 + 1.022)])
def test_absorbed_em_particle_deposits_everything(codec, pid, expect) -> None:
    """Nothing leaves: the balance has no outgoing energy to adjust, so the deposit is E_in (+ 2 m_e for a positron).
    A flow E_dep below that loses energy in every photoabsorption -- 6 MeV per 1 GeV e- shower in a W/Si chain."""
    pen, dp = codec
    f = _event(pen, dp, 0.5, 0.2, [])
    _run(pen, dp, f, 0.5, pid)
    edep, _ = _decode(pen, dp, f, 0.5)
    assert edep == pytest.approx(expect, rel=1e-4)


def test_absorbed_hadron_keeps_its_deposit(codec) -> None:
    """A captured neutron's balance is open (binding energy): its deposit stays the flow's."""
    pen, dp = codec
    f = _event(pen, dp, 0.5, 0.2, [])
    _run(pen, dp, f, 0.5, 2112)
    edep, _ = _decode(pen, dp, f, 0.5)
    assert edep == pytest.approx(0.2, rel=1e-4)


def test_incoming_at_the_cutoff_is_left_alone(codec) -> None:
    """At E_in = cutoff the outgoing encoding is degenerate (log ratio 0/0): no NaN, no change."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 4.0, [(11, 700.0), (22, 300.0)])
    before = f.clone()
    _run(pen, dp, f, CUT, 11)
    assert torch.equal(before.nan_to_num(), f.nan_to_num())


def test_hadron_survivor_loses_exactly_its_deposit(codec) -> None:
    """A proton the flow left at E_in (zero loss, clamped) leaves with E_in - E_dep - others."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 3.0, [(2212, 1000.0), (2112, 0.5)])
    _run(pen, dp, f, 1000.0, 2212)
    _, e = _decode(pen, dp, f, 1000.0)
    assert float(e[0]) == pytest.approx(996.5, abs=1e-2)
    assert float(e[1]) == pytest.approx(0.5, rel=1e-4)


def test_em_only_survivor_closes_at_any_share(codec) -> None:
    """A 1 GeV proton crossing the 108.8 mm W/Si stack keeps 75-79% of E_in; when everything else out is a gamma /
    e+- Geant4's balance is exact (residual 0.00 MeV up to q90, 37.8% of events) -- the 90% rule never fired there."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 230.0, [(2212, 700.0), (22, 2.0)])
    _run(pen, dp, f, 1000.0, 2212)
    _, e = _decode(pen, dp, f, 1000.0)
    assert float(e[0]) == pytest.approx(768.0, abs=1e-2)
    assert float(e[1]) == pytest.approx(2.0, rel=1e-4)


def test_nuclear_survivor_below_the_share_is_left_alone(codec) -> None:
    """A neutron out marks a nuclear interaction, whose balance is open (Geant4 residual median 100 MeV)."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 180.0, [(2212, 700.0), (2112, 50.0)])
    before = f.clone()
    _run(pen, dp, f, 1000.0, 2212)
    assert torch.equal(before.nan_to_num(), f.nan_to_num())


def test_survivor_with_too_large_a_deposit_leaves_at_the_cutoff(codec) -> None:
    """E_dep + others > E_in leaves the leader a negative budget: it leaves at the cutoff (to_scalar_e's clamp)."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 990.0, [(2212, 760.0), (22, 20.0)])
    _run(pen, dp, f, 1000.0, 2212)
    _, e = _decode(pen, dp, f, 1000.0)
    assert e.isfinite().all() and float(e[0]) == pytest.approx(CUT, rel=1e-3)


def test_nuclear_interaction_is_left_alone(codec) -> None:
    """No outgoing proton keeps > 90% of E_in: Geant4's balance is not closed there (fragments, binding)."""
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 50.0, [(2112, 400.0), (211, 300.0)])
    before = f.clone()
    _run(pen, dp, f, 1000.0, 2212)
    assert torch.equal(before.nan_to_num(), f.nan_to_num())


def test_edep_cap_covers_every_gun_species_above_geant4() -> None:
    """GenerateOut caps E_dep at E_in + excess for every gun species, never below the largest excess Geant4 shows
    on the 0.1 MeV plates (n 2.3 MeV: capture; pi+ 70.5, pi- 142.5: the pion mass), so an unbounded neutron or
    pion deposit cannot create energy in a chain."""
    from legofmt.main.generate import _EDEP_EXCESS_MEV
    g4_max = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.001, 2112: 2.263, 211: 70.513, -211: 142.498}
    assert all(_EDEP_EXCESS_MEV.get(pid, -1.0) >= m - 1e-3 for pid, m in g4_max.items())


def test_padding_stays_padding(codec) -> None:
    pen, dp = codec
    f = _event(pen, dp, 1000.0, 4.0, [(11, 700.0), (22, 300.0)])
    _run(pen, dp, f, 1000.0, 11)
    assert _F(f).out_p[0, 2:, :7].isnan().all() and (_F(f).out_p[0, 2:, 7] == 0).all()


def _exit_event(rows: list[tuple[list[float], list[float]]], L: int = 12) -> torch.Tensor:
    """(1, L, 8) event whose outgoing rows carry (momentum direction, position direction), pdgid 11."""
    set_layout(COND)
    f = torch.full((1, L, 8), float("nan"))
    f[..., 7] = 0.0
    for i, (m, u) in enumerate(rows):
        _F(f).out_p[0, i, 1:4] = torch.nn.functional.normalize(torch.tensor(m), dim=0)
        _F(f).out_p[0, i, 4:7] = torch.nn.functional.normalize(torch.tensor(u), dim=0)
        _F(f).out_p[0, i, 0] = 0.5
        _F(f).out_p[0, i, 7] = 11
    return f


def test_exit_momentum_is_mirrored_out_of_its_face() -> None:
    """A particle leaving through a face moves away from it (Geant4: never otherwise). One the flow sends back into
    its exit face would enter the next slab moving away from it -- the count model's out-of-distribution input."""
    from legofmt.main.generate import _exit_outward
    f = _exit_event([([0.6, 0.0, -0.8], [0.1, 0.2, 50.0]),    # +z face, moving -z: mirrored
                     ([0.6, 0.0, 0.8], [0.1, 0.2, 50.0]),     # +z face, moving +z: kept
                     ([-0.5, 0.5, 0.7], [1.0, 0.2, 0.3]),     # +x face, moving -x: mirrored
                     ([0.0, 0.3, -0.9], [0.2, 0.1, -50.0])])  # -z face, moving -z: kept
    before = f.clone()
    _exit_outward(f)
    m = _F(f).out_p[0, :4, 1:4]
    expect = _F(before).out_p[0, :4, 1:4].clone()
    expect[0, 2] *= -1
    expect[2, 0] *= -1
    assert torch.allclose(m, expect)
    assert torch.equal(_F(f).out_p[0, :4, [0, 4, 5, 6, 7]], _F(before).out_p[0, :4, [0, 4, 5, 6, 7]])
    assert _F(f).out_p[0, 4:, :7].isnan().all()
