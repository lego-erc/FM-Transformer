"""A fired pass-through token must bypass the flow and emit the primary unchanged."""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import _F, set_layout
from legofmt.main.generate import GenerateOut
from test_helpers import _gun_cond, _pt_mult_config, _save_flow_ckpt, _save_mult_ckpt


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("pt_ckpts")
    torch.manual_seed(0)
    fp = _save_flow_ckpt(tmp)
    torch.manual_seed(0)
    return GenerateOut(str(fp), str(_save_mult_ckpt(tmp, _pt_mult_config(passthrough_head=True))), device="cpu")


@pytest.fixture(autouse=True)
def _layout(generator: GenerateOut) -> None:
    set_layout(generator.cond_names)     # other modules leave their own layout behind


def _force(gen: GenerateOut, value: float) -> None:
    with torch.no_grad():
        gen.gen_mult.pt_head[-1].bias.fill_(value)


def test_fired_events_emit_the_primary_unchanged_and_deposit_nothing(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    sols, mask, attn = generator(_gun_cond(22, batch=6))
    assert (mask.sum(-1) == 0).all()
    assert _F(sols).edep.abs().max().item() == 0.0
    inc, out = _F(sols).in_p, _F(sols).out_p
    occupied = attn[:, generator.n_prefix + 1:].bool()
    assert occupied.sum(-1).eq(1).all()
    first = out[:, 0]
    assert first[:, 0].abs().max().item() == 0.0            # e_out == e_in
    assert torch.allclose(first[:, 1:7], inc[:, 0, 1:7])    # same direction
    assert first[:, -1].eq(22).all()                        # same species


def test_a_charged_primary_never_fires(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    _, mask, _ = generator(_gun_cond(2212, batch=6))
    assert (mask.sum(-1) > 0).all()


def test_a_suppressed_token_routes_everything_through_the_flow(generator: GenerateOut) -> None:
    """No row may short-circuit, and the flow must actually have run on them."""
    _force(generator, -50.0)
    torch.manual_seed(4)
    sols, mask, _ = generator(_gun_cond(22, batch=6))
    assert (mask.sum(-1) > 0).all()
    # a short-circuited row would still carry the zeroed E_dep gen_batch wrote
    assert _F(sols).edep.abs().max().item() > 0.0


def test_passthrough_prob_is_the_head_probability(generator: GenerateOut) -> None:
    """A slab chain fast-forwards a neutral particle through every plate it would pass: it needs P(pass) itself, the
    probability the head's Bernoulli draw uses -- 0 for species the head does not cover."""
    _force(generator, 50.0)
    assert torch.allclose(generator.passthrough_prob(_gun_cond(22, batch=6)), torch.ones(6))
    assert generator.passthrough_prob(_gun_cond(2212, batch=6)).abs().max().item() == 0.0
    _force(generator, 0.0)
    p = generator.passthrough_prob(_gun_cond(22, batch=4000))
    torch.manual_seed(7)
    _, mask, _ = generator(_gun_cond(22, batch=4000))
    assert abs((mask.sum(-1) == 0).float().mean().item() - p.mean().item()) < 0.03


def test_passthrough_off_samples_the_interacting_branch(generator: GenerateOut) -> None:
    """At the plate a fast-forwarded particle interacts in, the model must not pass it through again: the count
    decoder and the flow are both trained on the interacting events only."""
    _force(generator, 50.0)
    _, mask, _ = generator(_gun_cond(22, batch=6), passthrough=False)
    assert (mask.sum(-1) > 0).all()


def _no_outgoing(gen: GenerateOut, monkeypatch) -> list[int]:
    """Make the count model emit nothing, and record how many rows reach the flow per call."""
    real = gen.gen_mult.forward
    monkeypatch.setattr(gen.gen_mult, "forward", lambda *a, **k: torch.zeros_like(real(*a, **k)))
    rows: list[int] = []
    flow = gen.model.forward
    monkeypatch.setattr(gen.model, "forward", lambda batch, *a, **k: rows.append(len(batch[0])) or flow(batch, *a, **k))
    return rows


def test_absorbed_em_events_skip_the_flow_under_energy_balance(generator: GenerateOut, monkeypatch) -> None:
    """With energy_balance an e+-/gamma that leaves nothing deposits the balance budget, overwriting whatever E_dep the
    flow drew: 37% of the flow rows of a 1 GeV e- slab chain. Those events skip the flow; a hadron with no outgoing
    particle keeps its generated deposit, so it still needs it."""
    rows = _no_outgoing(generator, monkeypatch)
    monkeypatch.setitem(generator.model.rc.odeint_conf, "energy_balance", True)
    cond = torch.cat((_gun_cond(22, batch=4), _gun_cond(211, batch=3)))
    sols, mask, _ = generator(cond, passthrough=False)
    assert rows == [3]
    nc = generator.n_cond
    e_in = cond[:4, nc:nc + 3].norm(dim=-1)                   # the momentum column is energy-scaled
    edep = generator.dp.denorm_edep(_F(sols).edep[:4].double())
    torch.testing.assert_close(edep, e_in.double(), rtol=1e-4, atol=1e-6)
    assert (mask[:4].sum(-1) == 1).all()               # still an interacting event: the E_dep row is generated


def test_without_energy_balance_every_interacting_event_is_flowed(generator: GenerateOut, monkeypatch) -> None:
    rows = _no_outgoing(generator, monkeypatch)
    monkeypatch.setitem(generator.model.rc.odeint_conf, "energy_balance", False)
    generator(_gun_cond(22, batch=4), passthrough=False)
    assert rows == [4]
