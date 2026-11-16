"""``GenerateOut`` caps the generated E_dep at the incoming kinetic energy for primaries
that cannot deposit more (e-, gamma, p; e+ plus the 1.022 MeV its annihilation
releases) and leaves pi+- and n alone, whose captures release rest mass / binding
energy (stack59 500k test set: pi+- exceed E_in in 14-17% of events, by up to 178 MeV).
The flow's E_dep row has no wall at E_in -- it is normalised by ``max_energy``."""

from __future__ import annotations

import pytest
import torch

from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_helpers import _fill_edep, _gun_cond, _save_flow_ckpt, _save_mult_ckpt


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("ceiling_ckpts")
    return GenerateOut(str(_save_flow_ckpt(tmp, LEGOLtng)), str(_save_mult_ckpt(tmp)), device="cpu")


@torch.no_grad()
@pytest.mark.parametrize("pdgid, capped", [(22, True), (2212, True), (211, False)])
def test_edep_is_capped_at_the_incoming_energy(generator, monkeypatch, pdgid, capped) -> None:
    # a flow that puts every event's E_dep at the channel ceiling (max_energy = 300 MeV), twice E_in
    monkeypatch.setattr(generator.model, "forward", _fill_edep(1.0))
    dp = DataPrep(generator.model.rc.config)
    e_in = 150.0
    sols, _, _ = generator(_gun_cond(pdgid, e_in, batch=64))
    edep = dp.denorm_edep(_F(sols).edep)
    flowed = edep > 0                         # pass-through events never reach the flow and keep E_dep = 0
    assert flowed.any()
    expect = e_in if capped else 300.0
    assert torch.allclose(edep[flowed], torch.full_like(edep[flowed], expect), rtol=1e-4), edep


@torch.no_grad()
def test_zero_deposit_sentinel_is_untouched(generator, monkeypatch) -> None:
    monkeypatch.setattr(generator.model, "forward", _fill_edep(-0.1))   # the overflow_delta sentinel: no deposit
    sols, _, _ = generator(_gun_cond(22))
    assert (_F(sols).edep <= 0).all()
