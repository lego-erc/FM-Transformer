"""The kinetic-energy channel (``energy_kin``, the default).

The energy channel and the cutoff use the kinetic energy Geant4 records in
column 0 of a raw particle. With ``energy_kin: false`` (legacy) they use the
momentum norm, which saturates for hadrons (a proton of 1 GeV kinetic energy
has |p| = 1696 MeV > max_energy) and silently clamps 11.8% of the outgoing
particles.
"""

from __future__ import annotations

import math

import pytest
import torch

from test_helpers import MANIFOLD, _prep_config
from legofmt.data.write.cut import GetLEGOData
from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import _F
from legofmt.geometry.energy_proj import EnergyProjections

CUTOFF, MAX_E = 10.0, 1000.0


def _p_of(t: float, m: float) -> float:
    return math.sqrt(t * (t + 2 * m))


def _scal(e: float) -> float:
    return math.log(e / CUTOFF) / math.log(MAX_E / CUTOFF)


def _raw_token(t: float, m: float, direction, pos, pdgid: float):
    """Returns a raw particle ``[E_kin, p (3), x (3), pdgid]`` as pyg4lego records it."""
    p = _p_of(t, m) if m > 0 else t
    return [t, *[p * d for d in direction], *pos, pdgid]


def _prep(energy_kin: bool | None) -> DataPrep:
    cfg = _prep_config(max_energy=MAX_E, energy_kin=energy_kin)
    if energy_kin is None:
        del cfg["energy_kin"]
    return DataPrep(cfg)


def _batch():
    # incoming proton T=100 (|p|=444.5), one outgoing neutron T=20 (|p|=194.7)
    cc = torch.tensor([[
        _raw_token(100.0, 938.27, (1, 0, 0), (-50, 0, 0), 2212),
        _raw_token(20.0, 939.57, (0, 1, 0), (0, 50, 0), 2112),
    ]])
    mask = torch.tensor([[0, 1]])
    attn = torch.ones(1, 2, dtype=torch.bool)
    add = {"E_dep": torch.tensor([50.0]), "Density": torch.tensor([3.0])}
    return cc, mask, attn, add


def _prepped(energy_kin: bool | None):
    """Returns the tokens as a file stores them and as ``norm_e`` loads them: ``prep`` keeps the
    channels that depend on ``max_energy`` in MeV, ``norm_e`` puts them on the model scale."""
    prep = _prep(energy_kin)
    stored = prep.prep(_batch())
    return stored[0], prep.norm_e(stored)[0]


@pytest.mark.parametrize("energy_kin", [True, None], ids=["explicit", "default"])
def test_energy_kin_scalar_from_kinetic_column(energy_kin) -> None:
    """The energy channel holds the kinetic energy, with the key and without it: kinetic is the default."""
    stored, loaded = _prepped(energy_kin)
    assert abs(float(_F(stored).in_p[..., 0, 0]) - 100.0) < 1e-4      # MeV in the file
    got = float(_F(loaded).in_p[..., 0, 0])
    assert abs(got - _scal(100.0)) < 1e-5, (got, _scal(100.0))
    # the outgoing energy is stored in MeV too; loaded on the legacy log scale, it is 1 - s_out / s_in of the kinetic
    # energies: log(max_energy / cutoff) cancels in the ratio, so the channel does not depend on max_energy
    assert abs(float(_F(stored).out_p[..., 0, 0]) - 20.0) < 1e-4
    want = 1 - _scal(20.0) / _scal(100.0)
    got_o = float(_F(loaded).out_p[..., 0, 0])
    assert abs(got_o - want) < 1e-5, (got_o, want)


def test_explicit_false_is_momentum_based() -> None:
    stored, loaded = _prepped(energy_kin=False)
    p = _p_of(100.0, 938.27)
    assert abs(float(_F(stored).in_p[..., 0, 0]) - p) < 1e-3
    got = float(_F(loaded).in_p[..., 0, 0])
    assert abs(got - _scal(p)) < 1e-5, got


def test_energy_kin_exothermic_outgoing_clamps_to_zero() -> None:
    # a capture gamma with T=30, above the incoming neutron's T=12: stored as it is; on the legacy log scale the ratio
    # clamps at 1, and the channel is 0
    cc = torch.tensor([[
        _raw_token(12.0, 939.57, (1, 0, 0), (-50, 0, 0), 2112),
        _raw_token(30.0, 0.0, (0, 1, 0), (0, 50, 0), 22),
    ]])
    batch = (cc, torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool),
             {"E_dep": torch.tensor([1.0]), "Density": torch.tensor([3.0])})
    prep = _prep(energy_kin=True)
    stored = prep.prep(batch)
    assert float(_F(stored[0]).out_p[..., 0, 0]) == pytest.approx(30.0)
    assert float(_F(prep.norm_e(stored)[0]).out_p[..., 0, 0]) == 0.0


def test_energy_kin_cutoff_uses_kinetic() -> None:
    # neutron T=5 (|p|=97): kept by the momentum cutoff, dropped by the kinetic one
    inc = torch.tensor([[_raw_token(500.0, 938.27, (1, 0, 0), (-50, 0, 0), 2212)]])
    out = torch.tensor([[_raw_token(5.0, 939.57, (0, 1, 0), (0, 50, 0), 2112),
                         _raw_token(50.0, 0.0, (0, 0, 1), (0, 0, 50), 22)]])
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
    from legofmt.config import resolve_legoltng_config

    (tmp_path / "meta.json").write_text(json.dumps({
        "ntokens": 8, "particles": [11.0, 22.0, 2212.0],
        "max_energy": MAX_E, "cutoff_mev": CUTOFF,
        "cond_scalars": ["Density"], "energy_kin": True,
    }))
    cfg = {
        "dl_conf": {"lds_args": {"data": str(tmp_path), "cutoff_mev": CUTOFF}, "bs": 2},
        "model_conf": {
            "manifold": MANIFOLD,
            "max_energy": MAX_E, "base_pretrain_batches": 0,
            "model_args": {"h_dim": 8, "in_dim": 7, "nlayers": 1, "nhead": 1},
        },
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
    }
    # resolve_legoltng_config deep-copies its input, so the flag has to be
    # checked on the resolved config, which is what scripts/train.py saves
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


def test_mult_energy_rescale_is_the_same_physical_energy() -> None:
    """``GenerateOut`` normalises ``cond`` with the flow's ``max_energy``, so a multiplicity
    model trained at another ``max_energy`` needs its energy column rescaled. That is a pure
    change of the logarithm's base: the MeV it decodes to must not change."""
    cut, max_flow, max_mult = 10.0, 2000.0, 1000.0
    p_flow = EnergyProjections(cutoff_mev=cut, max_energy=max_flow)
    p_mult = EnergyProjections(cutoff_mev=cut, max_energy=max_mult)
    scale = float(torch.tensor(max_flow / cut).log() / torch.tensor(max_mult / cut).log())
    e = torch.tensor([10.0, 100.0, 548.0, 1000.0])
    assert torch.allclose(
        (p_flow.to_scalar_e(e) * scale).clamp(0.0, 1.0), p_mult.to_scalar_e(e), atol=1e-6,
    )
    # the failure it prevents: 1 GeV read with the other max_energy is 548 MeV
    off = p_flow.to_scalar_e(torch.tensor([1000.0]))
    assert abs(float(p_mult.cutoff * (max_mult / cut) ** off) - 548.0) < 2.0
