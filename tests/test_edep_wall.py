"""model_conf.edep_wall: the E_dep ROW encoded as the log-odds of the deposited fraction of a per-species cap.

    E_cap = E_in + margin(species),  f = E_dep / E_cap,  z = log E_dep - log(E_cap - E_dep)
    u = (z - z_lo) / (z_hi - z_lo),  z_lo = log(eps / E_cap),  z_hi = log(E_cap / eps_missing)

Small deposits keep the old log scale (z ~ log f); near the cap the channel resolves the MISSING energy, so a
contained shower's narrow peak below E_in becomes wide, and E_dep = E_cap is the channel's edge u = 1. The per-layer
tensor keeps the old elementwise scale. Off by default: every other model is unchanged.
"""

from __future__ import annotations

import copy

import pytest
import torch

from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F

from test_helpers import _fake_batch, _tiny_config


def _prep(wall: bool = True) -> DataPrep:
    cfg = copy.deepcopy(_tiny_config()["config"])
    cfg["model_conf"].update(edep_log_min=1e-6, edep_wall=wall, max_energy=2000.0)
    return DataPrep(cfg)


def test_round_trip_below_the_cap() -> None:
    dp = _prep()
    e_in = torch.tensor([1000.0, 300.0, 50.0])
    for pid in (11, 2212, 211):
        p = torch.full_like(e_in, pid)
        cap = e_in + {11: 0.0, 2212: 0.0, 211: 200.0}[pid]
        for frac in (1e-6, 1e-3, 0.3, 0.9, 0.99, 0.999):
            E = cap * frac
            back = dp.denorm_edep_row(dp.norm_edep_row(E, e_in, p), e_in, p)
            torch.testing.assert_close(back, E, rtol=2e-4, atol=0)


def test_zero_is_zero_and_the_cap_is_the_edge() -> None:
    dp = _prep()
    e_in, pid = torch.tensor([1000.0, 1000.0, 1000.0]), torch.tensor([11.0, 11.0, 11.0])
    u = dp.norm_edep_row(torch.tensor([0.0, 1000.0, 1300.0]), e_in, pid)
    assert u[0] == 0 and u[1] == 1 and u[2] == 1                     # zero stays the sentinel's 0; >= cap -> edge
    E = dp.denorm_edep_row(torch.tensor([-0.1, 1.0, 1.7]), e_in, pid)
    assert E[0] == 0 and (E[1:] <= 1000.0).all() and (E[1:] > 999.99).all()


def test_the_contained_peak_is_wide_in_the_new_channel() -> None:
    """The 1 GeV e- peak (0.96 - 0.995 E_in) is ~0.0013 wide on the old log row and should span > 0.05 here."""
    wall, old = _prep(), _prep(wall=False)
    e_in, pid = torch.tensor([1000.0, 1000.0]), torch.tensor([11.0, 11.0])
    E = torch.tensor([960.0, 995.0])
    u = wall.norm_edep_row(E, e_in, pid)
    assert (u[1] - u[0]) > 0.05
    u_old = old.norm_edep_row(E, e_in, pid)
    assert (u_old[1] - u_old[0]) < 0.002


def test_small_deposits_keep_a_log_scale() -> None:
    dp = _prep()
    e_in, pid = torch.full((3,), 1000.0), torch.full((3,), 2212.0)
    u = dp.norm_edep_row(torch.tensor([1e-3, 1e-2, 1e-1]), e_in, pid)
    torch.testing.assert_close(u[1] - u[0], u[2] - u[1], rtol=1e-3, atol=0)   # equal steps per decade


def test_pion_margin_allows_absorption_above_e_in() -> None:
    dp = _prep()
    e_in, pid = torch.tensor([500.0]), torch.tensor([211.0])
    E = torch.tensor([640.0])                                         # 140 MeV above E_in: pion absorption
    assert dp.norm_edep_row(E, e_in, pid) < 1
    torch.testing.assert_close(dp.denorm_edep_row(dp.norm_edep_row(E, e_in, pid), e_in, pid), E, rtol=2e-4, atol=0)


def test_without_the_wall_the_row_is_the_old_scale() -> None:
    dp = _prep(wall=False)
    E, e_in, pid = torch.tensor([0.0, 3.0, 900.0]), torch.full((3,), 1000.0), torch.full((3,), 11.0)
    torch.testing.assert_close(dp.norm_edep_row(E, e_in, pid), dp.norm_edep(E))
    torch.testing.assert_close(dp.denorm_edep_row(dp.norm_edep(E), e_in, pid), dp.denorm_edep(dp.norm_edep(E)))


@pytest.mark.parametrize("wall", [True, False])
def test_norm_e_encodes_the_row_from_the_raw_incoming_energy(wall: bool) -> None:
    """Loading a post-energy-split file: the incoming row holds E_in in MeV and the E_dep row MeV; norm_e must
    encode the row with E_in BEFORE it rewrites the incoming energy as its scalar. Layers keep the old scale."""
    dp = _prep(wall)
    ds = _fake_batch()
    f = ds.f.full.clone()
    _F(f).in_cc[..., 0, 0] = 1000.0
    _F(f).in_p[..., 0, -1] = 11.0
    _F(f).edep[:] = torch.tensor([980.0, 4.0])
    layers = torch.tensor([[500.0, 480.0], [3.0, 1.0]])
    out_f, _, _, out_layers = dp.norm_e((f, ds.m.full, ds.am.full, layers))
    expect = dp.norm_edep_row(torch.tensor([980.0, 4.0]), torch.full((2,), 1000.0), torch.full((2,), 11.0))
    torch.testing.assert_close(_F(out_f).edep, expect)
    torch.testing.assert_close(out_layers, dp.norm_edep(layers))


@torch.no_grad()
def test_generate_out_keeps_the_wall_row_and_skips_the_mev_clamp(tmp_path, monkeypatch) -> None:
    """With the wall the row is bounded by the cap by construction: GenerateOut must not apply the old MeV clamp
    (which works on the old scale and would rewrite u), and the row decodes to <= E_cap."""
    from legofmt.main.generate import GenerateOut
    from legofmt.main.modules import LEGOLtng
    from test_edep_ceiling import _cond
    from test_helpers import _save_flow_ckpt, _save_mult_ckpt

    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng)), str(_save_mult_ckpt(tmp_path)), device="cpu")
    gen.dp.edep_wall = True

    def at_the_edge(batch, *_):
        f, m, am = batch
        f = f.clone()
        _F(f).edep.fill_(1.0)
        return f, m, am
    monkeypatch.setattr(gen.model, "forward", at_the_edge)
    sols, _, _ = gen(_cond(22, 150.0, batch=64))
    row = _F(sols).edep
    flowed = row > 0
    assert flowed.any() and (row[flowed] == 1.0).all()
    E = gen.dp.denorm_edep_row(row, torch.full_like(row, 150.0), torch.full_like(row, 22.0))
    assert (E[flowed] <= 150.0).all() and (E[flowed] > 149.99).all()
