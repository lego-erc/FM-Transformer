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
from test_helpers import _save_flow_ckpt, _save_mult_ckpt


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("ceiling_ckpts")
    return GenerateOut(str(_save_flow_ckpt(tmp, LEGOLtng)), str(_save_mult_ckpt(tmp)), device="cpu")


def _cond(pdgid: int, e_mev: float, batch: int = 4) -> torch.Tensor:
    """``[B, 8]``: density, E-scaled mom along +z, entry pos on the -z face, pdgid."""
    cond = torch.zeros(batch, 8)
    cond[:, 0] = 1.0
    cond[:, 3] = e_mev
    cond[:, 6] = -1.0
    cond[:, 7] = pdgid
    return cond


@torch.no_grad()
@pytest.mark.parametrize("pdgid, capped", [(22, True), (2212, True), (211, False)])
def test_edep_is_capped_at_the_incoming_energy(generator, monkeypatch, pdgid, capped) -> None:
    # a flow that puts every event's E_dep at the channel ceiling (max_energy = 300 MeV), twice E_in
    def overflowing(batch, *_):
        f, m, am = batch
        f = f.clone()
        _F(f).edep.fill_(1.0)
        return f, m, am
    monkeypatch.setattr(generator.model, "forward", overflowing)
    dp = DataPrep(generator.model.rc.config)
    e_in = 150.0
    sols, _, _ = generator(_cond(pdgid, e_in, batch=64))
    edep = dp.denorm_edep(_F(sols).edep)
    flowed = edep > 0                         # pass-through events never reach the flow and keep E_dep = 0
    assert flowed.any()
    expect = e_in if capped else 300.0
    assert torch.allclose(edep[flowed], torch.full_like(edep[flowed], expect), rtol=1e-4), edep


@torch.no_grad()
def test_zero_deposit_sentinel_is_untouched(generator, monkeypatch) -> None:
    def absorbed(batch, *_):
        f, m, am = batch
        f = f.clone()
        _F(f).edep.fill_(-0.1)   # the overflow_delta sentinel: no deposit
        return f, m, am
    monkeypatch.setattr(generator.model, "forward", absorbed)
    sols, _, _ = generator(_cond(22, 150.0))
    assert (_F(sols).edep <= 0).all()
