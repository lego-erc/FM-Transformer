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


def _prep(wall: bool = True, species=None) -> DataPrep:
    cfg = copy.deepcopy(_tiny_config()["config"])
    cfg["model_conf"].update(edep_log_min=1e-6, edep_wall=wall, max_energy=2000.0)
    if species is not None:
        cfg["model_conf"]["edep_wall_species"] = species
    return DataPrep(cfg)


def test_wall_species_keeps_the_log_row_for_the_others() -> None:
    """edep_wall_species [11, -11, 22]: e-/e+/gamma rows take the wall, every other species the log row -- the wall
    resolves a contained EM shower near E_in but its prior misplaces a hadron's deposit."""
    some, every, none = _prep(species=[11, -11, 22]), _prep(), _prep(wall=False)
    e_in, E = torch.full((4,), 1000.0), torch.tensor([980.0, 980.0, 225.0, 225.0])
    pid = torch.tensor([11.0, 22.0, 2212.0, 2112.0])
    u = some.norm_edep_row(E, e_in, pid)
    torch.testing.assert_close(u[:2], every.norm_edep_row(E, e_in, pid)[:2])
    torch.testing.assert_close(u[2:], none.norm_edep_row(E, e_in, pid)[2:])
    torch.testing.assert_close(some.denorm_edep_row(u, e_in, pid), E, rtol=2e-4, atol=0)
    assert some.edep_wall_mask(pid).tolist() == [True, True, False, False]
    assert every.edep_wall_mask(pid).all() and not none.edep_wall_mask(pid).any()


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


@torch.no_grad()
def test_g4_api_hands_out_the_log_scale_row(tmp_path, monkeypatch) -> None:
    """Callers of gen_model_w_g4_args (lego-eval) decode its E_dep with the checkpoint's edep_log_min, i.e. as the
    log row: a wall model re-encodes its row on that scale there, or every MeV they report is wrong."""
    from legofmt.main.generate import GenerateOut
    from legofmt.main.modules import LEGOLtng
    from test_helpers import _save_flow_ckpt, _save_mult_ckpt

    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng)), str(_save_mult_ckpt(tmp_path)), device="cpu")
    gen.dp.edep_wall = True
    e_in, pid, E = 150.0, 22, 140.0
    u = float(gen.dp.norm_edep_row(torch.tensor([E]), torch.tensor([e_in]), torch.tensor([float(pid)])))

    def deposits(batch, *_):
        f, m, am = batch
        f = f.clone()
        _F(f).edep.fill_(u)
        return f, m, am
    monkeypatch.setattr(gen.model, "forward", deposits)
    out = gen.gen_model_w_g4_args(16, torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[0.0, 0.0, 1.0]]),
                                  torch.tensor([e_in]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([pid]))
    row = out["per_event"]["E_dep"]
    assert row.shape == out["per_particle"]["Outgoing"].shape[:1]      # one value per event
    flowed = row > 0
    assert flowed.any()
    torch.testing.assert_close(gen.dp.denorm_edep(row[flowed]), torch.full_like(row[flowed], E), rtol=1e-3, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("pid, clamped", [(22, False), (2212, True)])
def test_the_mev_clamp_applies_to_the_log_row_species_only(tmp_path, monkeypatch, pid, clamped) -> None:
    """With edep_wall_species [11, -11, 22] a photon's wall row is capped by construction and left alone, while a
    proton's log row is still clamped at E_in (+ its _EDEP_EXCESS_MEV, 0 for protons)."""
    from legofmt.main.generate import GenerateOut
    from legofmt.main.modules import LEGOLtng
    from test_edep_ceiling import _cond
    from test_helpers import _save_flow_ckpt, _save_mult_ckpt

    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng)), str(_save_mult_ckpt(tmp_path)), device="cpu")
    gen.dp.edep_wall, gen.dp.edep_wall_species = True, [11, -11, 22]

    def at_the_edge(batch, *_):
        f, m, am = batch
        f = f.clone()
        _F(f).edep.fill_(1.0)
        return f, m, am
    monkeypatch.setattr(gen.model, "forward", at_the_edge)
    sols, _, _ = gen(_cond(pid, 150.0, batch=64))
    row = _F(sols).edep
    flowed = row > 0
    assert flowed.any()
    if clamped:
        torch.testing.assert_close(row[flowed], gen.dp.norm_edep(torch.full_like(row[flowed], 150.0)))
    else:
        assert (row[flowed] == 1.0).all()


# --- data-driven margin: model_conf.edep_excess_mev ---------------------------------------------------------------
# Per incoming species, the largest E_dep - E_in [MeV] in the training data (meta.json edep_excess_max_mev, written by
# gen_data.py; for an older dataset computed from data_prepped.pt at fresh training). It replaces the hand-set
# EDEP_WALL_MARGIN_MEV / _EDEP_EXCESS_MEV; checkpoints without it keep the constants.

def _prep_excess(excess: dict, species=None) -> DataPrep:
    cfg = copy.deepcopy(_tiny_config()["config"])
    cfg["model_conf"].update(edep_log_min=1e-6, edep_wall=True, max_energy=2000.0, edep_excess_mev=excess)
    if species is not None:
        cfg["model_conf"]["edep_wall_species"] = species
    return DataPrep(cfg)


def _raw_file(e_in, pid, edep) -> torch.Tensor:
    """A stored (pre-norm_e) tensor: incoming kinetic energy and E_dep in MeV."""
    f = _fake_batch(B=len(e_in)).f.full.clone()
    _F(f).in_cc[..., 0, 0] = torch.tensor(e_in)
    _F(f).in_p[..., 0, -1] = torch.tensor(pid)
    _F(f).edep.copy_(torch.tensor(edep).view_as(_F(f).edep))
    return f


def test_edep_excess_max_is_the_largest_deposit_above_e_in_per_species() -> None:
    _prep()                                                           # sets the row layout _F reads
    f = _raw_file([100.0, 100.0, 300.0, 300.0], [-211.0, -211.0, 11.0, 11.0], [240.0, 90.0, 290.0, 299.0])
    assert DataPrep.edep_excess_max(f) == {-211: 140.0, 11: 0.0}               # never below 0: E_in itself is reachable


def test_the_cap_is_e_in_plus_the_data_excess() -> None:
    """A neutron deposit 2 MeV above E_in is the edge when the data's largest excess is 2 MeV; the hand-set 10 MeV
    margin left it inside the channel."""
    dp, const = _prep_excess({2112: 2.0, 211: 150.0}), _prep()
    e_in, pid = torch.tensor([500.0, 500.0]), torch.tensor([2112.0, 211.0])
    E = torch.tensor([502.0, 640.0])
    u = dp.norm_edep_row(E, e_in, pid)
    assert u[0] == 1 and const.norm_edep_row(E, e_in, pid)[0] < 1
    torch.testing.assert_close(dp.denorm_edep_row(u, e_in, pid)[1], E[1], rtol=2e-4, atol=0)


def test_a_walled_species_missing_from_the_data_is_an_error() -> None:
    with pytest.raises(ValueError, match="edep_excess_mev"):
        _prep_excess({11: 0.0}, species=[11, 22])


def test_fresh_training_takes_the_excess_from_meta_or_the_data(tmp_path) -> None:
    import json
    from legofmt.mod_comps.config import resolve_legoltng_config
    from test_helpers import MANIFOLD

    meta = {"ntokens": 8, "particles": [11.0, 22.0, 2212.0], "max_energy": 1000.0, "cutoff_mev": 10.0,
            "cond_scalars": ["Density"], "energy_kin": True}
    cfg = {"dl_conf": {"lds_args": {"data": str(tmp_path), "cutoff_mev": 10.0}, "bs": 2},
           "model_conf": {"manifold": MANIFOLD, "max_energy": 1000.0, "base_pretrain_batches": 0, "edep_wall": True,
                          "model_args": {"h_dim": 8, "in_dim": 7, "nlayers": 1, "nhead": 1}},
           "opt_conf": {"opt": "schedulefree", "lr": 1e-3}}
    _prep()
    f = _raw_file([100.0, 300.0], [2112.0, 22.0], [104.5, 250.0])
    torch.save((f, torch.ones(f.shape[:2]), torch.ones(f.shape[:2], dtype=torch.bool)), tmp_path / "data_prepped.pt")
    (tmp_path / "meta.json").write_text(json.dumps(meta))             # a dataset from before gen_data wrote the key
    assert resolve_legoltng_config(cfg).config["model_conf"]["edep_excess_mev"] == {2112: 4.5, 22: 0.0}
    meta["edep_excess_max_mev"] = {"11": 0.0, "22": 0.0, "2212": 0.001}   # json keys are strings
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    rc = resolve_legoltng_config(cfg)
    assert rc.config["model_conf"]["edep_excess_mev"] == {11: 0.0, 22: 0.0, 2212: 0.001}


@torch.no_grad()
def test_the_mev_clamp_uses_the_data_excess(tmp_path, monkeypatch) -> None:
    """A log-row proton is clamped at E_in + its data excess (30 MeV here), not at the hand-set E_in + 0."""
    from legofmt.main.generate import GenerateOut
    from legofmt.main.modules import LEGOLtng
    from test_edep_ceiling import _cond
    from test_helpers import _save_flow_ckpt, _save_mult_ckpt

    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng)), str(_save_mult_ckpt(tmp_path)), device="cpu")
    gen.dp.edep_excess_mev = {2212: 30.0}

    def at_the_edge(batch, *_):
        f, m, am = batch
        f = f.clone()
        _F(f).edep.fill_(1.0)
        return f, m, am
    monkeypatch.setattr(gen.model, "forward", at_the_edge)
    sols, _, _ = gen(_cond(2212, 150.0, batch=64))
    row = _F(sols).edep
    flowed = row > 0
    assert flowed.any()
    torch.testing.assert_close(row[flowed], gen.dp.norm_edep(torch.full_like(row[flowed], 180.0)))
