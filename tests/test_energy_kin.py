"""Tests for the kinetic-energy channel (``energy_kin``).

With ``energy_kin: true`` the energy scalar and the cutoff use the RAW kinetic
energy Geant4 records in column 0 -- not the momentum norm, which saturates for
hadrons (a 1 GeV-kinetic proton has |p| = 1696 MeV > max_energy) and silently
clamps 11.8%% of outgoing particles. Default stays momentum-based.
"""

from __future__ import annotations

import math

import torch

from legofmt.data.dataloaders import GetLEGOData
from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F

CUTOFF, MAX_E = 10.0, 1000.0


def _p_of(t: float, m: float) -> float:
    return math.sqrt(t * (t + 2 * m))


def _scal(e: float) -> float:
    return math.log(e / CUTOFF) / math.log(MAX_E / CUTOFF)


def _raw_row(t: float, m: float, direction, pos, pdgid: float):
    """[E_kin, Px, Py, Pz, X, Y, Z, PDGID] as pyg4lego records it."""
    p = _p_of(t, m) if m > 0 else t
    return [t, *[p * d for d in direction], *pos, pdgid]


def _prep(energy_kin: bool | None) -> DataPrep:
    cfg = {
        "cutoff_mev": CUTOFF, "max_energy": MAX_E, "proj_ray": False,
        "cond_scalars": ["Density"],
        "manifold": [{"name": "euclidean", "dim": 1},
                     {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}],
    }
    if energy_kin is not None:
        cfg["energy_kin"] = energy_kin
    return DataPrep(cfg)


def _batch():
    # incoming proton T=100 (|p|=444.5), one outgoing neutron T=20 (|p|=194.7)
    cc = torch.tensor([[
        _raw_row(100.0, 938.27, (1, 0, 0), (-50, 0, 0), 2212),
        _raw_row(20.0, 939.57, (0, 1, 0), (0, 50, 0), 2112),
    ]])
    mask = torch.tensor([[0, 1]])
    attn = torch.ones(1, 2, dtype=torch.bool)
    add = {"E_dep": torch.tensor([50.0]), "Density": torch.tensor([3.0])}
    return cc, mask, attn, add


def _prepped(energy_kin: bool | None):
    """(stored file, loaded-and-normalised) -- prep() keeps the max_energy-dependent
    channels in MeV; DataPrep.norm_e puts them on the model scale at load time."""
    prep = _prep(energy_kin)
    stored = prep.prep(_batch())
    return stored[0], prep.norm_e(stored)[0]


def test_energy_kin_scalar_from_kinetic_column() -> None:
    stored, loaded = _prepped(energy_kin=True)
    assert abs(float(_F(stored).in_p[..., 0, 0]) - 100.0) < 1e-4      # MeV in the file
    got = float(_F(loaded).in_p[..., 0, 0])
    assert abs(got - _scal(100.0)) < 1e-5, (got, _scal(100.0))
    # outgoing stores 1 - s_out/s_in in KINETIC scalars, and needs no conversion:
    # log_range cancels in the ratio, so the column is max_energy-independent
    want = 1 - _scal(20.0) / _scal(100.0)
    for tag, f in (("stored", stored), ("loaded", loaded)):
        got_o = float(_F(f).out_p[..., 0, 0])
        assert abs(got_o - want) < 1e-5, (tag, got_o, want)


def test_default_is_kinetic() -> None:
    stored, loaded = _prepped(energy_kin=None)
    assert abs(float(_F(stored).in_p[..., 0, 0]) - 100.0) < 1e-4
    got = float(_F(loaded).in_p[..., 0, 0])
    assert abs(got - _scal(100.0)) < 1e-5, got


def test_explicit_false_is_momentum_based() -> None:
    stored, loaded = _prepped(energy_kin=False)
    p = _p_of(100.0, 938.27)
    assert abs(float(_F(stored).in_p[..., 0, 0]) - p) < 1e-3
    got = float(_F(loaded).in_p[..., 0, 0])
    assert abs(got - _scal(p)) < 1e-5, got


def test_energy_kin_exothermic_outgoing_clamps_to_zero() -> None:
    # capture gamma T=30 above the incoming neutron's T=12: ratio clamps -> 0
    cc = torch.tensor([[
        _raw_row(12.0, 939.57, (1, 0, 0), (-50, 0, 0), 2112),
        _raw_row(30.0, 0.0, (0, 1, 0), (0, 50, 0), 22),
    ]])
    batch = (cc, torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool),
             {"E_dep": torch.tensor([1.0]), "Density": torch.tensor([3.0])})
    f, _, _ = _prep(energy_kin=True).prep(batch)
    assert float(_F(f).out_p[..., 0, 0]) == 0.0


def test_energy_kin_cutoff_uses_kinetic() -> None:
    # neutron T=5 (|p|=97): kept by the momentum cutoff, dropped by the kinetic one
    inc = torch.tensor([[_raw_row(500.0, 938.27, (1, 0, 0), (-50, 0, 0), 2212)]])
    out = torch.tensor([[_raw_row(5.0, 939.57, (0, 1, 0), (0, 50, 0), 2112),
                         _raw_row(50.0, 0.0, (0, 0, 1), (0, 0, 50), 22)]])
    data = {"per_particle": {"Incoming": inc, "Outgoing": out},
            "per_event": {"E_dep": torch.tensor([1.0]), "Density": torch.tensor([3.0])}}
    for kin, want_n in ((False, 2), (True, 1)):
        pp, _, attn, _ = GetLEGOData(cutoff_mev=CUTOFF, energy_kin=kin)(data)
        assert int(attn.sum()) - 1 == want_n, (kin, attn)
    # default = kinetic
    pp, _, attn, _ = GetLEGOData(cutoff_mev=CUTOFF)(data)
    assert int(attn.sum()) - 1 == 1, attn


# --- the channel must be recorded in the model config ---------------------
# Nothing downstream can tell a kinetic-channel checkpoint from a |p|-channel
# one by inspection, and confusing them silently halves hadron momenta at
# decode time. meta.json knows; the resolved config has to carry it forward so
# it lands in the saved checkpoint.

def test_energy_kin_is_carried_from_meta_into_model_conf(tmp_path) -> None:
    import json
    from legofmt.mod_comps.config import resolve_legoltng_config

    (tmp_path / "meta.json").write_text(json.dumps({
        "ntokens": 8, "particles": [11.0, 22.0, 2212.0],
        "max_energy": MAX_E, "cutoff_mev": CUTOFF,
        "cond_scalars": ["Density"], "energy_kin": True,
    }))
    cfg = {
        "dl_conf": {"lds_args": {"data": str(tmp_path), "cutoff_mev": CUTOFF}, "bs": 2},
        "model_conf": {
            "manifold": [{"name": "euclidean", "dim": 1},
                         {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}],
            "max_energy": MAX_E, "base_pretrain_batches": 0,
            "model_args": {"h_dim": 8, "in_dim": 7, "nlayers": 1, "nhead": 1},
        },
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
    }
    # resolve_legoltng_config deep-copies its input, so the flag has to be
    # checked on the resolved config -- which is what scripts/train.py saves
    # into the checkpoint (`model.rc.config`).
    rc = resolve_legoltng_config(cfg)
    assert rc.config["model_conf"].get("energy_kin") is True, (
        "energy_kin from meta.json did not reach the resolved model_conf; the "
        "saved checkpoint cannot describe its own energy channel"
    )


# --- E_dep normalisation --------------------------------------------------
# E_dep is normalised on load by DataPrep.norm_e using the *model's*
# max_energy, never baked into the stored dataset.

def test_edep_is_normalised_by_model_max_energy() -> None:
    stored, loaded = _prepped(energy_kin=None)          # E_dep = 50
    raw = float(_F(stored).edep.flatten()[0])
    assert abs(raw - 50.0) < 1e-6, f"the file should keep E_dep in MeV, got {raw}"
    got = float(_F(loaded).edep.flatten()[0])
    assert abs(got - 50.0 / MAX_E) < 1e-6, f"E_dep should be 50/{MAX_E}, got {got}"
