"""``model_conf.out_margin``: the outgoing energy as the log-odds of its fraction of a cap per pair of species.

```
cap = e_in + margin(pid_in, pid_out)       z    = logit(e_out / cap)
e   = 1 - (z - z_lo) / (z_hi - z_lo)       z_lo = log(cutoff / cap),   z_hi = log(cap / 1e-3 MeV)
```

The E_dep token's margin scale (``test_edep_margin.py``) turned round, so that 0 is still the cap and 1 the cutoff.
Near the cap it resolves the energy a surviving particle lost, and the margin lets a decay or capture product carry
more than E_in. A dataset stores the outgoing energy in MeV, and ``norm_e`` encodes it when the dataset is loaded; a
file from before stores the legacy log scale, which only a model without the key can read.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from legofmt.data.load.packed import load_dataset
from legofmt.data.struct import _F, set_layout
from legofmt.data.write.norm import DataPrep
from legofmt.data.write.pack import pack
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_helpers import (MANIFOLD, PACKED_COND, _gun_cond, _save_flow_ckpt, _save_mult_ckpt, packed_prep,
                          prepped_padded, write_two_dirs)

CUT, E_IN = 0.1, 1000.0
# pi+ decay at rest: nu_mu 29.8 MeV, then the mu+ decay's e+ up to 52.8 MeV
TABLE = {211: {14: 29.8, -11: 52.8, 22: 0.0}, 11: {22: 0.0}}


def _prep(margin: bool = True, table=TABLE, cutoff: float = CUT) -> DataPrep:
    return DataPrep({"cutoff_mev": cutoff, "manifold": MANIFOLD, "proj_ray": False, "cond_scalars": ["Density"],
                     "max_energy": 2000.0, "out_margin": margin, "out_excess_mev": table})


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values)


def _scale(x: float, cap: float) -> float:
    """The scale as the module docstring writes it, in float64."""
    z, z_lo, z_hi = math.log(x / (cap - x)), math.log(CUT / cap), math.log(cap / 1e-3)
    return 1 - (z - z_lo) / (z_hi - z_lo)


# ---- the scale ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lost", [0.3, 3.3, 30.0])
def test_a_survivor_is_resolved_near_the_cap(lost: float) -> None:
    """A surviving e- keeps its species, so its cap is E_in. The energy it lost (0.3 / 3.3 / 30 MeV of 1 GeV) lands at
    e = 0.25 / 0.35 / 0.45, where the legacy log scale put it within 0.004 of 0."""
    x = _t(E_IN - lost)
    e = _prep().norm_out_energy(x, _t(E_IN), _t(11), _t(11))
    assert float(e) == pytest.approx(_scale(float(x), E_IN), abs=1e-6)
    assert float(_prep(margin=False).norm_out_energy(x, _t(E_IN), _t(11), _t(11))) < 0.004


def test_the_cap_gives_0_and_the_cutoff_almost_1() -> None:
    e = _prep().norm_out_energy(_t(E_IN, CUT), _t(E_IN, E_IN), _t(11, 11), _t(11, 22))
    assert float(e[0]) == 0.0
    assert float(e[1]) > 0.999


def test_decoding_inverts_encoding() -> None:
    """In float64 the round trip keeps the energy lost to 1e-6 of itself, down to 0.01 MeV below the cap."""
    x = torch.tensor([0.1, 1.0, 10.0, 500.0, 990.0, 999.0, 999.9, 999.99], dtype=torch.float64)
    e_in, pid = torch.full_like(x, E_IN), torch.full_like(x, 11.0)
    dp = _prep()
    back = dp.denorm_out_energy(dp.norm_out_energy(x, e_in, pid, pid), e_in, pid, pid)
    torch.testing.assert_close(E_IN - back, E_IN - x, rtol=1e-6, atol=0.0)


def test_a_decay_product_above_the_incoming_energy_keeps_its_energy() -> None:
    """A pi+ of 10 MeV decays at rest, and its nu_mu carries 29.8 MeV. The pair's margin puts the cap above it, so the
    energy survives the round trip; the legacy scale held it at E_in (e = 0)."""
    x, e_in, pid_in, pid_out = _t(29.8), _t(10.0), _t(211), _t(14)
    dp = _prep()
    e = dp.norm_out_energy(x, e_in, pid_in, pid_out)
    assert 0.0 < float(e) < 1.0
    assert float(dp.denorm_out_energy(e, e_in, pid_in, pid_out)) == pytest.approx(29.8, rel=1e-5)
    assert float(_prep(margin=False).norm_out_energy(x, e_in, pid_in, pid_out)) == 0.0


def test_a_particle_of_the_incoming_species_is_capped_at_the_incoming_energy() -> None:
    """The margin of a pair of one species is 0, whatever the table holds."""
    dp = _prep(table={**TABLE, 2212: {2212: 50.0}})
    assert float(dp.norm_out_energy(_t(1000.5), _t(E_IN), _t(2212), _t(2212))) == 0.0


def test_a_pair_missing_from_the_table_takes_the_widest_margin_of_its_incoming_species() -> None:
    """211 -> 2112 is not in the table: 211's widest margin, 52.8 MeV, holds. An incoming species missing from it takes
    the widest margin of all; 11's widest is 0, which caps 11 -> 2112 at E_in."""
    dp = _prep()
    for pid_in in (211.0, 2212.0):
        assert float(dp.norm_out_energy(_t(60.0), _t(10.0), _t(pid_in), _t(2112))) > 0.0   # cap 62.8 MeV
    assert float(dp.norm_out_energy(_t(11.0), _t(10.0), _t(11), _t(2112))) == 0.0


def test_without_the_key_the_legacy_log_scale() -> None:
    """``e = 1 - log(e_out / cutoff) / log(e_in / cutoff)``, held in [0, 1]: what a dataset stored before."""
    x = torch.tensor([0.1, 5.0, 500.0, 1000.0, 1200.0])
    e_in = torch.full_like(x, E_IN)
    want = [1 - min(1.0, max(0.0, math.log(v / CUT) / math.log(E_IN / CUT))) for v in x.tolist()]
    got = _prep(margin=False, table=None).norm_out_energy(x, e_in, torch.full_like(x, 11.0), torch.full_like(x, 22.0))
    assert got.tolist() == pytest.approx(want, abs=1e-6)


def test_the_key_needs_the_margins() -> None:
    with pytest.raises(ValueError, match="out_excess_mev"):
        _prep(table=None)


# ---- the margins: the data's largest excess per pair ----------------------------------------------------------------

def test_the_margins_are_the_data_maximum_per_pair() -> None:
    """Per pair of incoming and outgoing species, the largest ``e_out - e_in`` of a stored file, 0 where no particle
    exceeds E_in. A pair of one species and the padding (pdgid 0) do not count."""
    set_layout(("Density",))
    f = torch.zeros(3, 6, 8)   # [Density, E_dep, incoming, 3 outgoing], every energy in MeV
    events = [(211, 10.0, [(14, 29.8), (-11, 40.0), (211, 12.0)]),
              (211, 10.0, [(14, 25.0), (22, 3.0), (0, 1.0)]),
              (11, 100.0, [(22, 60.0), (0, 1.0), (0, 1.0)])]
    for i, (pid, e_in, outgoing) in enumerate(events):
        f[i, 2, 0], f[i, 2, -1] = e_in, pid
        for k, (q, e) in enumerate(outgoing):
            f[i, 3 + k, 0], f[i, 3 + k, -1] = e, q
    got = DataPrep.out_excess_max(f)
    assert set(got) == {211, 11}
    assert got[211] == pytest.approx({14: 19.8, -11: 30.0, 22: 0.0}, abs=1e-5)
    assert got[11] == {22: 0.0}


# ---- the dataset: stored in MeV, encoded when loaded ----------------------------------------------------------------

def _mev_dir(tmp_path):
    """Writes a dataset directory as ``scripts/gen_data.py`` does: the packed events in MeV, and meta.json with the
    margins. Returns the stored tensors and the directory."""
    t = prepped_padded(PACKED_COND)
    d = tmp_path / "mev"
    d.mkdir()
    torch.save(pack(*t, n_prefix=len(PACKED_COND) + 1).to_dict(), d / "data_packed.pt")
    margins = DataPrep.out_excess_max(t[0])
    (d / "meta.json").write_text(json.dumps({
        "ntokens": t[0].shape[1], "particles": [22.0, 2212.0], "particles_in": [2212.0], "cutoff_mev": 10.0,
        "cond_scalars": list(PACKED_COND), "energy_kin": True, "max_energy": 300.0, "format": "packed",
        "n_events": len(t[0]), "out_excess_max_mev": {p: row for p, row in margins.items()}}))
    return t, d, margins


@pytest.mark.parametrize("margin", [True, False], ids=["margin", "legacy"])
def test_a_dataset_in_mev_is_encoded_when_loaded(tmp_path, margin: bool) -> None:
    t, d, margins = _mev_dir(tmp_path)
    prep = packed_prep(out_margin=True, out_excess_mev=margins) if margin else packed_prep()
    loaded = load_dataset(str(d), prep=prep).data
    stored, real = _F(t[0]), loaded.am.out_p.bool()
    want = prep.norm_out_energy(stored.out_energy, stored.in_energy, stored.in_p[..., -1], stored.out_pid)
    torch.testing.assert_close(_F(loaded.f.full).out_energy[real], want[real])
    assert (_F(loaded.f.full).out_energy[real] <= 1.0).all()


def test_norm_e_encodes_what_prep_writes() -> None:
    """``prep`` writes the outgoing energy in MeV, and ``norm_e`` on its padded tensors encodes it; padding stays."""
    t = prepped_padded(PACKED_COND)
    prep = packed_prep()
    f = prep.norm_e(t)[0]
    stored, real = _F(t[0]), t[2][:, len(PACKED_COND) + 2:]
    want = prep.norm_out_energy(stored.out_energy, stored.in_energy, stored.in_p[..., -1], stored.out_pid)
    torch.testing.assert_close(_F(f).out_energy[real], want[real])
    assert torch.equal(_F(f).out_energy[~real], stored.out_energy[~real])


def test_a_dataset_from_before_cannot_take_the_margin_scale(tmp_path) -> None:
    """A file from before stores the legacy log scale, clamped at E_in: the MeV that the margin scale needs are gone."""
    pad, pk, _ = write_two_dirs(tmp_path)
    prep = packed_prep(out_margin=True, out_excess_mev={2212: {22: 0.0}})
    for d in (pad, pk):
        with pytest.raises(ValueError, match="out_margin"):
            load_dataset(str(d), prep=prep)


def test_a_fresh_model_takes_the_margins_from_meta_json(tmp_path) -> None:
    """meta.json's ``out_excess_max_mev`` (string keys) becomes ``model_conf.out_excess_mev`` (pdgids), and the
    training data loads on the margin scale."""
    _, d, margins = _mev_dir(tmp_path)
    model = LEGOLtng(_fresh_config(d, out_margin=True))
    assert model.rc.config["model_conf"]["out_excess_mev"] == margins
    model.setup()
    assert len(model._train_ds) + len(model._val_ds) == 6


def _fresh_config(data_dir, **model_conf) -> dict:
    """Returns a fresh flow config on the dataset ``data_dir``, with the ``model_conf`` keys set."""
    return {"dl_conf": {"lds_args": {"data": str(data_dir), "cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
            "val_conf": {"val_frac": 0.34, "seed": 0},
            "base_conf": {"kappa": 8.0, "base_dist": "poles", "scale_dist": "uniform"},
            "model_conf": {"manifold": MANIFOLD, "max_energy": 300.0,
                           "model_args": {"h_dim": 16, "nlayers": 1, "nhead": 2, "in_dim": 7, "ff_mult": 1,
                                          "dropout": 0.0},
                           **model_conf},
            "opt_conf": {"opt": "schedulefree", "lr": 1e-3}}


# ---- generation -----------------------------------------------------------------------------------------------------

GEN_TABLE = {22: {211: 5.0}}   # the tiny flow's species: 22, 211, 2212


@torch.no_grad()
def test_generated_energies_decode_between_the_scale_ends(tmp_path) -> None:
    """``dp.denorm_out_energy`` turns a generated outgoing energy into MeV: at most the pair's cap, and at least
    ``cap cutoff / (cap + cutoff)``, which e = 1 decodes to."""
    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng, out_margin=True, out_excess_mev=GEN_TABLE)),
                      str(_save_mult_ckpt(tmp_path)), device="cpu")
    out = gen.gen_model_w_g4_args(64, torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[0.0, 0.0, 1.0]]),
                                  torch.tensor([150.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([22]))
    o = out["per_particle"]["Outgoing"]
    pid = o[..., -1]
    E = gen.dp.denorm_out_energy(o[..., 0], torch.full_like(pid[:, :1], 150.0), torch.full_like(pid[:, :1], 22.0), pid)
    cap = 150.0 + 5.0 * (pid == 211)
    real = pid != 0
    assert real.any()
    assert ((E <= cap) & (E >= cap * 10.0 / (cap + 10.0) * (1 - 1e-6)))[real].all()


@torch.no_grad()
def test_the_legacy_energy_balance_refuses_the_margin_scale(tmp_path) -> None:
    """``odeint_conf.energy_balance`` decodes the outgoing energy on the legacy log scale."""
    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng, out_margin=True, out_excess_mev=GEN_TABLE)),
                      str(_save_mult_ckpt(tmp_path)), device="cpu")
    gen.model.rc.odeint_conf["energy_balance"] = True
    with pytest.raises(ValueError, match="energy_balance"):
        gen(_gun_cond(22, batch=2))
