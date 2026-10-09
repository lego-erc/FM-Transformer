"""The pass-through gate: a primary that traverses without interacting.

The
species channel only resolves at ``t = 1``, so a short-circuit cannot be read
off it -- the gate is its own small MLP over the conditioning, answered before
the solve. A fired event must skip the ODE entirely and emit the primary
verbatim, and must be weighted out of the flow's own losses.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import DataStruct, _F, set_layout
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_generate_direct import _save_flow_ckpt
from test_modules_direct import _tiny_config


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    torch.manual_seed(0)
    return GenerateOut(str(_save_flow_ckpt(tmp_path_factory.mktemp("gate"))), device="cpu")


def _force(gen: GenerateOut, value: float) -> None:
    with torch.no_grad():
        gen.model.pt_head[-1].bias.fill_(value)
        gen.model.pt_head[-1].weight.zero_()


def _cond(gen: GenerateOut, pdgid: int, batch: int = 6) -> torch.Tensor:
    cond = torch.zeros(batch, gen.n_cond + 7)
    cond[:, 0] = 1.0
    cond[:, gen.n_cond:gen.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, gen.n_cond + 3:gen.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, -1] = pdgid
    return cond


def test_fired_events_never_reach_the_solver(generator: GenerateOut) -> None:
    """The whole point of the gate: skip ~50 transformer forwards, not mask them."""
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    orig = generator.model.solve
    generator.model.solve = lambda *a, **k: pytest.fail("solver ran on a fired event")
    try:
        sols, mask, _ = generator(_cond(generator, 22))
    finally:
        generator.model.solve = orig
    assert (mask.sum(-1) == 0).all()
    assert _F(sols).edep.abs().max().item() == 0.0


def test_fired_events_emit_the_primary_unchanged(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    sols, _, attn = generator(_cond(generator, 22))
    inc, out = _F(sols).in_p, _F(sols).out_p
    occupied = attn[:, generator.n_prefix + 1:]
    assert occupied.sum(-1).eq(1).all()
    first = out[:, 0]
    assert first[:, 0].abs().max().item() == 0.0            # e_out == e_in
    assert torch.allclose(first[:, 1:7], inc[:, 0, 1:7])    # same direction
    assert first[:, -1].eq(22).all()                        # same species
    assert out[:, 1:, :7].isnan().all()                     # nothing else survives


def test_a_charged_primary_never_fires(generator: GenerateOut) -> None:
    """A charged particle always deposits something; the gate is masked to neutrals."""
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    _, mask, _ = generator(_cond(generator, 2212))
    assert (mask.sum(-1) > 0).all()


def test_a_suppressed_gate_routes_everything_through_the_flow(generator: GenerateOut) -> None:
    _force(generator, -50.0)
    set_layout(generator.cond_names)
    torch.manual_seed(4)
    sols, mask, _ = generator(_cond(generator, 22))
    assert (mask.sum(-1) > 0).all()
    assert _F(sols).edep.abs().max().item() > 0.0


def _pt_batch(B: int = 8, L: int = 5, passthrough: bool = True) -> DataStruct:
    """Every event a pass-through: E_dep 0, exactly one outgoing particle."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 0] = 0.0 if passthrough else 0.3     # E_dep
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 2, 0] = 0.8
    f[:, 2, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 2, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f[:, 2, 7] = 22.0
    f[:, 3, 0] = 0.8
    f[:, 3, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 3, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f[:, 3:, 7] = 22.0
    f[:, 4, :] = torch.nan                       # one outgoing particle only
    am = torch.ones(B, L, dtype=torch.bool)
    am[:, 4] = False
    m = am.clone().long()
    m[:, :3] = 0
    return DataStruct(f, m, am)


def test_passthrough_events_are_weighted_out_of_the_flow_loss() -> None:
    """With every event a pass-through, only the gate's BCE may contribute --
    the flow must not be asked to model a mode the gate already owns."""
    torch.manual_seed(0)
    model = LEGOLtng(_tiny_config())
    model.on_fit_start(); model.train()

    loss = model._step(_pt_batch(passthrough=True), 0)
    loss.backward()
    # the gate lives on vf so it rides the checkpoint, so exclude it by name here:
    # the claim is about the velocity field, which the gate is not part of
    flow_grad = sum(
        q.grad.abs().sum() for n, q in model.model.named_parameters()
        if q.grad is not None and "pt_head" not in n)
    gate_grad = sum(
        q.grad.abs().sum() for q in model.pt_head.parameters() if q.grad is not None)
    assert float(gate_grad) > 0, "gate did not train"
    assert float(flow_grad) == 0.0, f"flow trained on a pass-through event: {flow_grad}"


def test_an_interacting_batch_still_trains_the_flow() -> None:
    torch.manual_seed(0)
    model = LEGOLtng(_tiny_config())
    model.on_fit_start(); model.train()
    model._step(_pt_batch(passthrough=False), 0).backward()
    flow_grad = sum(
        p.grad.abs().sum() for p in model.model.parameters() if p.grad is not None)
    assert float(flow_grad) > 0


def test_the_gate_survives_a_checkpoint_round_trip(tmp_path) -> None:
    """``scripts/train.py`` saves only ``vf.state_dict()`` for the flow, so a gate
    living on the LightningModule reloads randomly initialised -- firing on roughly
    half of all neutral primaries. That is why ``pt_head`` sits on ``CFMTrafo_x``."""
    cfg = _tiny_config()["config"]
    a = LEGOLtng({"state_dict": {}, "config": cfg})
    with torch.no_grad():
        a.pt_head[-1].bias.fill_(3.14159)
    path = tmp_path / "flow.pt"
    torch.save({"state_dict": a.model.vf.state_dict(), "config": cfg}, path)

    b = LEGOLtng(torch.load(path, map_location="cpu", weights_only=False))
    assert torch.allclose(b.pt_head[-1].bias, a.pt_head[-1].bias), \
        "the pass-through gate did not survive the checkpoint"
