"""``model_conf.edep_margin``: the E_dep token as the log-odds of the deposited fraction of a cap per incoming species.

```
cap = e_in + margin(species)        z    = logit(edep / cap) = log(edep) - log(cap - edep)
u   = (z - z_lo) / (z_hi - z_lo)    z_lo = log(e_min / cap),   z_hi = log(cap / 1e-3 MeV),   e_min = edep_log_min
```

This is the margin scale (``DataPrep.norm_edep_margin``). A small deposit keeps a log scale (z ~ log(edep / cap)).
Near the cap the scale resolves the missing energy ``cap - edep``, so a contained shower's narrow peak below ``e_in``
becomes wide, and ``edep = cap`` is the edge ``u = 1``. The E_dep layers keep the layer scale
(``norm_edep_layer``), and so does the token of a model without the key.
"""

from __future__ import annotations

import json

import pytest
import torch

from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import _F
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from legofmt.config import resolve_legoltng_config

from test_helpers import MANIFOLD, _fake_batch, _fill_edep, _gun_cond, _save_flow_ckpt, _save_mult_ckpt, _tiny_config


def _prep(margin: bool = True, **model_conf) -> DataPrep:
    return DataPrep(_tiny_config(edep_log_min=1e-6, edep_margin=margin, max_energy=2000.0, **model_conf)["config"])


@pytest.fixture
def gen(tmp_path) -> GenerateOut:
    return GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng)), str(_save_mult_ckpt(tmp_path)), device="cpu")


def test_margin_species_keeps_the_log_token_for_the_others() -> None:
    """``edep_margin_species`` [11, -11, 22]: the token of an incoming e-/e+/gamma is on the margin scale, that of
    every other species on the layer scale. The margin scale resolves a contained EM shower near E_in, but its prior
    misplaces a hadron's deposit."""
    some, every, none = _prep(edep_margin_species=[11, -11, 22]), _prep(), _prep(margin=False)
    e_in, E = torch.full((4,), 1000.0), torch.tensor([980.0, 980.0, 225.0, 225.0])
    pid = torch.tensor([11.0, 22.0, 2212.0, 2112.0])
    u = some.norm_edep_margin(E, e_in, pid)
    torch.testing.assert_close(u[:2], every.norm_edep_margin(E, e_in, pid)[:2])
    torch.testing.assert_close(u[2:], none.norm_edep_margin(E, e_in, pid)[2:])
    torch.testing.assert_close(some.denorm_edep_margin(u, e_in, pid), E, rtol=2e-4, atol=0)
    assert some.edep_margin_mask(pid).tolist() == [True, True, False, False]
    assert every.edep_margin_mask(pid).all() and not none.edep_margin_mask(pid).any()


def test_round_trip_below_the_cap() -> None:
    dp = _prep()
    e_in = torch.tensor([1000.0, 300.0, 50.0])
    for pid in (11, 2212, 211):
        p = torch.full_like(e_in, pid)
        cap = e_in + {11: 0.0, 2212: 0.0, 211: 200.0}[pid]
        for frac in (1e-6, 1e-3, 0.3, 0.9, 0.99, 0.999):
            E = cap * frac
            back = dp.denorm_edep_margin(dp.norm_edep_margin(E, e_in, p), e_in, p)
            torch.testing.assert_close(back, E, rtol=2e-4, atol=0)


def test_zero_is_zero_and_the_cap_is_the_edge() -> None:
    dp = _prep()
    e_in, pid = torch.tensor([1000.0, 1000.0, 1000.0]), torch.tensor([11.0, 11.0, 11.0])
    u = dp.norm_edep_margin(torch.tensor([0.0, 1000.0, 1300.0]), e_in, pid)
    assert u[0] == 0 and u[1] == 1 and u[2] == 1                     # no deposit stays 0; >= cap -> the edge u = 1
    E = dp.denorm_edep_margin(torch.tensor([-0.1, 1.0, 1.7]), e_in, pid)
    assert E[0] == 0 and (E[1:] <= 1000.0).all() and (E[1:] > 999.99).all()


def test_the_contained_peak_is_wide_in_the_new_channel() -> None:
    """The 1 GeV e- peak (0.96 - 0.995 E_in) is ~0.0013 wide on the layer scale and spans > 0.05 on the margin scale."""
    new, old = _prep(), _prep(margin=False)
    e_in, pid = torch.tensor([1000.0, 1000.0]), torch.tensor([11.0, 11.0])
    E = torch.tensor([960.0, 995.0])
    u = new.norm_edep_margin(E, e_in, pid)
    assert (u[1] - u[0]) > 0.05
    u_old = old.norm_edep_margin(E, e_in, pid)
    assert (u_old[1] - u_old[0]) < 0.002


def test_small_deposits_keep_a_log_scale() -> None:
    dp = _prep()
    e_in, pid = torch.full((3,), 1000.0), torch.full((3,), 2212.0)
    u = dp.norm_edep_margin(torch.tensor([1e-3, 1e-2, 1e-1]), e_in, pid)
    torch.testing.assert_close(u[1] - u[0], u[2] - u[1], rtol=1e-3, atol=0)   # equal steps per decade


def test_pion_margin_allows_absorption_above_e_in() -> None:
    dp = _prep()
    e_in, pid = torch.tensor([500.0]), torch.tensor([211.0])
    E = torch.tensor([640.0])                                         # 140 MeV above E_in: pion absorption
    assert dp.norm_edep_margin(E, e_in, pid) < 1
    u = dp.norm_edep_margin(E, e_in, pid)
    torch.testing.assert_close(dp.denorm_edep_margin(u, e_in, pid), E, rtol=2e-4, atol=0)


def test_without_the_margin_the_token_is_the_old_scale() -> None:
    dp = _prep(margin=False)
    E, e_in, pid = torch.tensor([0.0, 3.0, 900.0]), torch.full((3,), 1000.0), torch.full((3,), 11.0)
    torch.testing.assert_close(dp.norm_edep_margin(E, e_in, pid), dp.norm_edep_layer(E))
    u = dp.norm_edep_layer(E)
    torch.testing.assert_close(dp.denorm_edep_margin(u, e_in, pid), dp.denorm_edep_layer(u))


@pytest.mark.parametrize("margin", [True, False])
def test_norm_e_encodes_the_token_from_the_raw_incoming_energy(margin: bool) -> None:
    """A stored file holds the incoming energy and the E_dep token in MeV. ``norm_e`` must encode the token with E_in
    before it rewrites the incoming energy on the model scale. The E_dep layers take the layer scale."""
    dp = _prep(margin)
    ds = _fake_batch()
    f = ds.f.full.clone()
    _F(f).in_cc[..., 0, 0] = 1000.0
    _F(f).in_p[..., 0, -1] = 11.0
    _F(f).edep[:] = torch.tensor([980.0, 4.0])
    layers = torch.tensor([[500.0, 480.0], [3.0, 1.0]])
    out_f, _, _, out_layers = dp.norm_e((f, ds.m.full, ds.am.full, layers))
    expect = dp.norm_edep_margin(torch.tensor([980.0, 4.0]), torch.full((2,), 1000.0), torch.full((2,), 11.0))
    torch.testing.assert_close(_F(out_f).edep, expect)
    torch.testing.assert_close(out_layers, dp.norm_edep_layer(layers))


@torch.no_grad()
def test_generate_out_keeps_the_margin_token_and_skips_the_mev_clamp(gen, monkeypatch) -> None:
    """A margin token is bounded by its cap by construction. ``GenerateOut`` must not apply the MeV clamp,
    which works on the layer scale and would rewrite ``u``; the token decodes to <= the cap."""
    gen.dp.edep_margin = True
    monkeypatch.setattr(gen.model, "forward", _fill_edep(1.0))
    sols, _, _ = gen(_gun_cond(22, batch=64))
    token = _F(sols).edep
    flowed = token > 0
    assert flowed.any() and (token[flowed] == 1.0).all()
    E = gen.dp.denorm_edep_margin(token, torch.full_like(token, 150.0), torch.full_like(token, 22.0))
    assert (E[flowed] <= 150.0).all() and (E[flowed] > 149.99).all()


@torch.no_grad()
def test_g4_api_hands_out_the_log_scale_token(gen, monkeypatch) -> None:
    """Callers of ``gen_model_w_g4_args`` (lego-eval) decode its E_dep with the checkpoint's ``edep_log_min``, i.e. on
    the layer scale. A margin model re-encodes its token on that scale there; otherwise every MeV they report is
    wrong."""
    gen.dp.edep_margin = True
    e_in, pid, E = 150.0, 22, 140.0
    u = float(gen.dp.norm_edep_margin(torch.tensor([E]), torch.tensor([e_in]), torch.tensor([float(pid)])))
    monkeypatch.setattr(gen.model, "forward", _fill_edep(u))
    out = gen.gen_model_w_g4_args(16, torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[0.0, 0.0, 1.0]]),
                                  torch.tensor([e_in]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([pid]))
    token = out["per_event"]["E_dep"]
    assert token.shape == out["per_particle"]["Outgoing"].shape[:1]      # one value per event
    flowed = token > 0
    assert flowed.any()
    torch.testing.assert_close(gen.dp.denorm_edep_layer(token[flowed]), torch.full_like(token[flowed], E),
                               rtol=1e-3, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("pid, clamped", [(22, False), (2212, True)])
def test_the_mev_clamp_applies_to_the_log_token_species_only(gen, monkeypatch, pid, clamped) -> None:
    """With ``edep_margin_species`` [11, -11, 22] a photon's margin token is capped by construction and left alone,
    while a proton's layer-scale token is still clamped at E_in (+ its ``HAND_SET_EXCESS_MEV``, 0 for protons)."""
    gen.dp.edep_margin, gen.dp.edep_margin_species = True, [11, -11, 22]
    monkeypatch.setattr(gen.model, "forward", _fill_edep(1.0))
    sols, _, _ = gen(_gun_cond(pid, batch=64))
    token = _F(sols).edep
    flowed = token > 0
    assert flowed.any()
    if clamped:
        torch.testing.assert_close(token[flowed], gen.dp.norm_edep_layer(torch.full_like(token[flowed], 150.0)))
    else:
        assert (token[flowed] == 1.0).all()


# --- data-driven margin: model_conf.edep_excess_mev ---------------------------------------------------------------
# Per incoming species, the largest E_dep - E_in [MeV] in the training data (meta.json edep_excess_max_mev, written by
# gen_data.py; for a dataset whose meta.json lacks the key, computed from data_prepped.pt when a fresh training
# starts). It takes the place of the hand-set HAND_SET_MARGIN_MEV / HAND_SET_EXCESS_MEV; a checkpoint without it keeps
# those constants.

def _raw_file(e_in, pid, edep) -> torch.Tensor:
    """Returns the tokens of a stored file (before ``norm_e``): the incoming kinetic energy and E_dep in MeV."""
    f = _fake_batch(B=len(e_in)).f.full.clone()
    _F(f).in_cc[..., 0, 0] = torch.tensor(e_in)
    _F(f).in_p[..., 0, -1] = torch.tensor(pid)
    _F(f).edep.copy_(torch.tensor(edep).view_as(_F(f).edep))
    return f


def test_edep_excess_max_is_the_largest_deposit_above_e_in_per_species() -> None:
    _prep()                                                           # sets the token layout _F reads
    f = _raw_file([100.0, 100.0, 300.0, 300.0], [-211.0, -211.0, 11.0, 11.0], [240.0, 90.0, 290.0, 299.0])
    assert DataPrep.edep_excess_max(f) == {-211: 140.0, 11: 0.0}               # never below 0: E_in itself is reachable


def test_the_cap_is_e_in_plus_the_data_excess() -> None:
    """A neutron deposit 2 MeV above E_in is the edge ``u = 1`` when the data's largest excess is 2 MeV; under the
    hand-set 10 MeV margin it lies below the edge."""
    dp, const = _prep(edep_excess_mev={2112: 2.0, 211: 150.0}), _prep()
    e_in, pid = torch.tensor([500.0, 500.0]), torch.tensor([2112.0, 211.0])
    E = torch.tensor([502.0, 640.0])
    u = dp.norm_edep_margin(E, e_in, pid)
    assert u[0] == 1 and const.norm_edep_margin(E, e_in, pid)[0] < 1
    torch.testing.assert_close(dp.denorm_edep_margin(u, e_in, pid)[1], E[1], rtol=2e-4, atol=0)


def test_the_old_key_names_still_select_the_margin_token() -> None:
    """Old configs and checkpoints name the keys ``edep_wall`` / ``edep_wall_species``."""
    conf = _tiny_config(edep_log_min=1e-6, edep_wall=True, edep_wall_species=[11], max_energy=2000.0)["config"]
    old = DataPrep(conf)
    assert old.edep_margin and old.edep_margin_species == [11]


def test_a_margin_species_missing_from_the_data_is_an_error() -> None:
    with pytest.raises(ValueError, match="edep_excess_mev"):
        _prep(edep_excess_mev={11: 0.0}, edep_margin_species=[11, 22])


def test_fresh_training_takes_the_excess_from_meta_or_the_data(tmp_path) -> None:
    meta = {"ntokens": 8, "particles": [11.0, 22.0, 2212.0], "max_energy": 1000.0, "cutoff_mev": 10.0,
            "cond_scalars": ["Density"], "energy_kin": True}
    cfg = {"dl_conf": {"lds_args": {"data": str(tmp_path), "cutoff_mev": 10.0}, "bs": 2},
           "model_conf": {"manifold": MANIFOLD, "max_energy": 1000.0, "base_pretrain_batches": 0, "edep_margin": True,
                          "model_args": {"h_dim": 8, "in_dim": 7, "nlayers": 1, "nhead": 1}},
           "opt_conf": {"opt": "schedulefree", "lr": 1e-3}}
    _prep()
    f = _raw_file([100.0, 300.0], [2112.0, 22.0], [104.5, 250.0])
    torch.save((f, torch.ones(f.shape[:2]), torch.ones(f.shape[:2], dtype=torch.bool)), tmp_path / "data_prepped.pt")
    (tmp_path / "meta.json").write_text(json.dumps(meta))             # a dataset whose meta.json lacks the key
    assert resolve_legoltng_config(cfg).config["model_conf"]["edep_excess_mev"] == {2112: 4.5, 22: 0.0}
    meta["edep_excess_max_mev"] = {"11": 0.0, "22": 0.0, "2212": 0.001}   # json keys are strings
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    rc = resolve_legoltng_config(cfg)
    assert rc.config["model_conf"]["edep_excess_mev"] == {11: 0.0, 22: 0.0, 2212: 0.001}


@torch.no_grad()
def test_the_mev_clamp_uses_the_data_excess(gen, monkeypatch) -> None:
    """A proton's layer-scale token is clamped at E_in + its data excess (30 MeV here), not at the hand-set E_in + 0."""
    gen.dp.edep_excess_mev = {2212: 30.0}
    monkeypatch.setattr(gen.model, "forward", _fill_edep(1.0))
    sols, _, _ = gen(_gun_cond(2212, batch=64))
    token = _F(sols).edep
    flowed = token > 0
    assert flowed.any()
    torch.testing.assert_close(token[flowed], gen.dp.norm_edep_layer(torch.full_like(token[flowed], 180.0)))
