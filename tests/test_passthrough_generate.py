"""An event whose pass-through flag is set must bypass the flow and return its incoming particle unchanged."""

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
    """Sets the bias of the pass-through head's output: at 50 every covered species passes through, at -50 none."""
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
    assert torch.allclose(first[:, 1:7], inc[:, 0, 1:7])    # same direction and position
    assert generator.pdgids_of(first[:, -1]).eq(22).all()   # same species


def test_count_model_is_in_eval_mode_from_construction(tmp_path) -> None:
    """``GenerateOut`` must put the multiplicity model in eval mode itself: left in training mode, the first call
    samples the pass-through with dropout on."""
    cfg = _pt_mult_config(passthrough_head=True)
    cfg["mm_conf"]["dropout"] = 0.5
    gen = GenerateOut(str(_save_flow_ckpt(tmp_path)), str(_save_mult_ckpt(tmp_path, cfg)), device="cpu")
    assert not gen.gen_mult.training


def test_a_charged_primary_never_fires(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    _, mask, _ = generator(_gun_cond(2212, batch=6))
    assert (mask.sum(-1) > 0).all()


def test_a_suppressed_token_routes_everything_through_the_flow(generator: GenerateOut) -> None:
    """With the pass-through flag suppressed no event may bypass the flow: the flow must have run on every one."""
    _force(generator, -50.0)
    torch.manual_seed(4)
    sols, mask, _ = generator(_gun_cond(22, batch=6))
    assert (mask.sum(-1) > 0).all()
    # an event that bypassed the flow would still carry the zero E_dep that gen_batch wrote
    assert _F(sols).edep.abs().max().item() > 0.0


def test_passthrough_prob_is_the_head_probability(generator: GenerateOut) -> None:
    """A slab chain fast-forwards a neutral particle through every plate it would pass. For that it needs the
    probability the head's Bernoulli sample uses, P(pass-through) itself: 0 for a species the head does not cover."""
    _force(generator, 50.0)
    assert torch.allclose(generator.passthrough_prob(_gun_cond(22, batch=6)), torch.ones(6))
    assert generator.passthrough_prob(_gun_cond(2212, batch=6)).abs().max().item() == 0.0
    _force(generator, 0.0)
    p = generator.passthrough_prob(_gun_cond(22, batch=4000))
    torch.manual_seed(7)
    _, mask, _ = generator(_gun_cond(22, batch=4000))
    assert abs((mask.sum(-1) == 0).float().mean().item() - p.mean().item()) < 0.03


def test_passthrough_off_samples_the_interacting_branch(generator: GenerateOut) -> None:
    """At the plate in which a fast-forwarded particle interacts, the model must not pass it through again: the
    counts and the flow are both trained on the interacting events only."""
    _force(generator, 50.0)
    _, mask, _ = generator(_gun_cond(22, batch=6), passthrough=False)
    assert (mask.sum(-1) > 0).all()


def _no_outgoing(gen: GenerateOut, monkeypatch) -> list[int]:
    """Makes the multiplicity model sample no outgoing particle; returns the list of how many events reach the flow
    in each call."""
    real = gen.gen_mult.forward
    monkeypatch.setattr(gen.gen_mult, "forward", lambda *a, **k: torch.zeros_like(real(*a, **k)))
    rows: list[int] = []
    flow = gen.model.forward
    monkeypatch.setattr(gen.model, "forward", lambda batch, *a, **k: rows.append(len(batch[0])) or flow(batch, *a, **k))
    return rows


def test_absorbed_em_events_skip_the_flow_under_energy_balance(generator: GenerateOut, monkeypatch) -> None:
    """With ``energy_balance`` an e+- / gamma without an outgoing particle deposits the balance budget, whatever E_dep
    the flow generates: 37 % of the events a 1 GeV e- slab chain sends to the flow. Those events skip the flow. A
    hadron without an outgoing particle keeps its generated E_dep, so it still runs through the flow."""
    rows = _no_outgoing(generator, monkeypatch)
    monkeypatch.setitem(generator.model.rc.odeint_conf, "energy_balance", True)
    cond = torch.cat((_gun_cond(22, batch=4), _gun_cond(211, batch=3)))
    sols, mask, _ = generator(cond, passthrough=False)
    assert rows == [3]
    nc = generator.n_cond
    e_in = cond[:4, nc:nc + 3].norm(dim=-1)                   # cond's momentum is energy-scaled
    edep = generator.dp.denorm_edep_layer(_F(sols).edep[:4].double())
    torch.testing.assert_close(edep, e_in.double(), rtol=1e-4, atol=1e-6)
    assert (mask[:4].sum(-1) == 1).all()               # still an interacting event: the E_dep token is generated


def test_without_energy_balance_every_interacting_event_is_flowed(generator: GenerateOut, monkeypatch) -> None:
    rows = _no_outgoing(generator, monkeypatch)
    monkeypatch.setitem(generator.model.rc.odeint_conf, "energy_balance", False)
    generator(_gun_cond(22, batch=4), passthrough=False)
    assert rows == [4]
