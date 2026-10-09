"""The species channel is generated, and with it the multiplicity.

Index 0 of the pdgid vocabulary already meant "empty slot", so generating the
species channel generates the occupancy: a slot that resolves back to 0 is not a
particle. These cover the training path (every padded slot is supervised), the
solver path (the class jumps along the trajectory), the pinning that lets a
caller condition on a known multiplicity, and ``GenerateOut`` with no
multiplicity checkpoint at all.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from legofmt.data.struct import DataStruct, _F, set_layout
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_generate_direct import _save_flow_ckpt
from test_modules_direct import _tiny_config


def _padded_batch(B: int = 8, L: int = 5) -> DataStruct:
    """2 conditioning slots, 1 incoming, ``L - 3`` outgoing -- of which the last
    is padding (NaN features, attn 0) in half the events. No other fixture in the
    suite carries real padding, and padding is the whole multiplicity signal."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 0] = 0.1
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 2, 0] = 0.8
    f[:, 2, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 2, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f[:, 2, 7] = 22.0
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 0] = torch.rand(B, L - 3, generator=g)
    f[:, 3:, 1:4] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 4:7] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 7] = 22.0

    am = torch.ones(B, L, dtype=torch.bool)
    am[: B // 2, -1] = False          # half the events are one particle short
    f[: B // 2, -1, :] = torch.nan    # padding is NaN, as dataset_cutoff writes it
    m = am.clone().long()
    m[:, :3] = 0
    return DataStruct(f, m, am)


def _model() -> LEGOLtng:
    model = LEGOLtng(_tiny_config())
    model.on_fit_start()
    return model


def test_step_is_finite_with_padding_and_trains_the_head() -> None:
    model = _model()
    model.train()
    loss = model._step(_padded_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()
    head = model.model.vf.species_head
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_pad_slots_are_supervised_not_skipped() -> None:
    """A padded slot must reach the species loss: its class-0 target is the only
    thing telling the model the event had fewer particles."""
    model = _model()
    model.train()
    seen = {}
    orig = torch.nn.functional.cross_entropy

    def spy(lg, tg, *a, **kw):
        seen["n"] = tg.numel()
        seen["zeros"] = int((tg == 0).sum())
        return orig(lg, tg, *a, **kw)

    torch.nn.functional.cross_entropy = spy
    try:
        model._step(_padded_batch(), 0)
    finally:
        torch.nn.functional.cross_entropy = orig
    assert seen["n"] == 8 * 2, "species loss did not cover every outgoing slot"
    assert seen["zeros"] == 4, "padded slots were not supervised as the empty class"


def test_solve_generates_species_in_vocabulary() -> None:
    model = _model()
    model.eval()
    ds = _padded_batch()
    m = ds.m.full.clone()
    m[:, 3:] = 1
    ds_g = DataStruct(ds.f.full.nan_to_num(1.0), m, torch.ones_like(ds.am.full))
    torch.manual_seed(0)
    x, sp = model.solve(ds_g, step_size=0.25, return_species=True, method="euler")
    npd = model.rc.model_args["npdgids"]
    assert x.shape == ds.f.model_in.shape
    assert sp.shape == ds.m.full.shape
    assert int(sp.min()) >= 0 and int(sp.max()) < npd
    # the conditioning rows keep the class they came in with
    assert torch.equal(sp[:, :3], model.convert_pdgids(ds_g.f.pdgids).squeeze(-1)[:, :3])


def test_a_prefilled_species_is_pinned() -> None:
    """Pre-assigning a class opts a slot out of generation -- this is what lets a
    caller condition on a known multiplicity (``gt_mult``)."""
    model = _model()
    model.eval()
    ds = _padded_batch()
    f = ds.f.full.nan_to_num(1.0)
    f[:, 2, 7] = 1.0                   # the column holds indices, not raw ids now
    f[:, 3:, 7] = 0.0
    f[:, 3, 7] = 1.0                   # slot 3 pinned to vocabulary index 1
    m = ds.m.full.clone()
    m[:, 3:] = 1
    model.rc = replace(model.rc, pdgid_is_idx=True)
    torch.manual_seed(0)
    _, sp = model.solve(
        DataStruct(f, m, torch.ones_like(ds.am.full)),
        step_size=0.25, return_species=True, method="euler",
    )
    assert (sp[:, 3] == 1).all(), f"pinned class drifted: {sp[:, 3]}"


def test_forward_reports_generated_occupancy() -> None:
    model = _model()
    model.eval()
    ds = _padded_batch()
    m = ds.m.full.clone()
    m[:, 3:] = 1
    ds_g = DataStruct(ds.f.full.nan_to_num(1.0), m, torch.ones_like(ds.am.full))
    torch.manual_seed(0)
    out, mask, am = model(ds_g)
    assert out.shape[-1] == 8
    # occupancy is an output: it must agree with the species column, and the
    # inactive slots must be NaN-filled
    sp = out[..., -1]
    assert torch.equal(am[:, 3:], sp[:, 3:] != 0)
    assert out[..., :7][~am].isnan().all()


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    torch.manual_seed(0)
    tmp = tmp_path_factory.mktemp("species_ckpts")
    return GenerateOut(str(_save_flow_ckpt(tmp)), device="cpu")


def test_generate_out_needs_no_multiplicity_checkpoint(generator: GenerateOut) -> None:
    set_layout(generator.cond_names)
    cond = torch.zeros(6, generator.n_cond + 7)
    cond[:, 0] = 1.0
    cond[:, generator.n_cond:generator.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, generator.n_cond + 3:generator.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, -1] = generator.pdgids[0].item()

    torch.manual_seed(0)
    sols, _, attn = generator(cond)
    assert generator.gen_mult is None
    out = _F(sols).out_p
    occupied = attn[:, generator.n_prefix + 1:]
    # the species column carries raw pdgids, and only on occupied slots
    assert torch.equal(occupied, out[..., -1] != 0)
    assert torch.isin(out[..., -1][occupied], generator.pdgids.to(out.dtype)).all()
    assert out[..., :7][~occupied].isnan().all()
