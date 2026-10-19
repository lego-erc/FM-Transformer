"""A fired pass-through token must bypass the flow and emit the primary unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from legofmt.data.struct import _F, set_layout
from legofmt.main.generate import GenerateOut
from legofmt.multiplicity.model import MultModel
from test_generate_direct import _save_flow_ckpt


def _mult_config() -> dict:
    return {
        "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
        "mm_conf": {
            "ptypes": torch.tensor([22, 211], dtype=torch.int64),
            "ptypes_in": torch.tensor([22, 211, 2212], dtype=torch.int64),
            "max_out_particles": 4,
            "max_count": 4,
            "h_dim": 16,
            "in_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "use_abs_pos_emb": False,
            "post_emb_norm": False,
            "train_inverse": False,
            "passthrough_head": True,
            "model_args": {"use_adaptive_rmsnorm": True},
        },
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
    }


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("pt_ckpts")
    torch.manual_seed(0)
    fp = _save_flow_ckpt(tmp)
    mcfg = _mult_config()
    torch.manual_seed(0)
    mult = MultModel({"state_dict": {}, "config": mcfg})
    mp = tmp / "mult.pt"
    torch.save({"state_dict": mult.state_dict(), "config": mcfg}, mp)
    return GenerateOut(str(fp), str(mp), device="cpu")


def _cond(gen: GenerateOut, pdgid: int, batch: int = 6) -> torch.Tensor:
    cond = torch.zeros(batch, gen.n_cond + 7)
    cond[:, 0] = 1.0
    cond[:, gen.n_cond:gen.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, gen.n_cond + 3:gen.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, -1] = pdgid
    return cond


def _force(gen: GenerateOut, value: float) -> None:
    with torch.no_grad():
        gen.gen_mult.pt_head[-1].bias.fill_(value)


def test_fired_events_carry_no_deposit(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    sols, mask, _ = generator(_cond(generator, 22))
    assert (mask.sum(-1) == 0).all()
    assert _F(sols).edep.abs().max().item() == 0.0


def test_fired_events_emit_the_primary_unchanged(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    sols, _, attn = generator(_cond(generator, 22))
    inc, out = _F(sols).in_p, _F(sols).out_p
    occupied = attn[:, generator.n_prefix + 1:].bool()
    assert occupied.sum(-1).eq(1).all()
    first = out[:, 0]
    assert first[:, 0].abs().max().item() == 0.0            # e_out == e_in
    assert torch.allclose(first[:, 1:7], inc[:, 0, 1:7])    # same direction
    assert first[:, -1].eq(22).all()                        # same species


def test_a_charged_primary_never_fires(generator: GenerateOut) -> None:
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    _, mask, _ = generator(_cond(generator, 2212))
    assert (mask.sum(-1) > 0).all()


def test_a_suppressed_token_routes_everything_through_the_flow(generator: GenerateOut) -> None:
    """No row may short-circuit, and the flow must actually have run on them."""
    _force(generator, -50.0)
    set_layout(generator.cond_names)
    torch.manual_seed(4)
    sols, mask, _ = generator(_cond(generator, 22))
    assert (mask.sum(-1) > 0).all()
    # a short-circuited row would still carry the zeroed E_dep gen_batch wrote
    assert _F(sols).edep.abs().max().item() > 0.0


def test_passthrough_prob_is_the_head_probability(generator: GenerateOut) -> None:
    """A slab chain fast-forwards a neutral particle through every plate it would pass: it needs P(pass) itself, the
    probability the head's Bernoulli draw uses -- 0 for species the head does not cover."""
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    assert torch.allclose(generator.passthrough_prob(_cond(generator, 22)), torch.ones(6))
    assert generator.passthrough_prob(_cond(generator, 2212)).abs().max().item() == 0.0
    _force(generator, 0.0)
    p = generator.passthrough_prob(_cond(generator, 22, batch=4000))
    torch.manual_seed(7)
    _, mask, _ = generator(_cond(generator, 22, batch=4000))
    assert abs((mask.sum(-1) == 0).float().mean().item() - p.mean().item()) < 0.03


def test_passthrough_off_samples_the_interacting_branch(generator: GenerateOut) -> None:
    """At the plate a fast-forwarded particle interacts in, the model must not pass it through again: the count
    decoder and the flow are both trained on the interacting events only."""
    _force(generator, 50.0)
    set_layout(generator.cond_names)
    _, mask, _ = generator(_cond(generator, 22), passthrough=False)
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
    set_layout(generator.cond_names)
    rows = _no_outgoing(generator, monkeypatch)
    monkeypatch.setitem(generator.model.rc.odeint_conf, "energy_balance", True)
    cond = torch.cat((_cond(generator, 22, batch=4), _cond(generator, 211, batch=3)))
    sols, mask, _ = generator(cond, passthrough=False)
    assert rows == [3]
    nc = generator.n_cond
    e_in = cond[:4, nc:nc + 3].norm(dim=-1)                   # the momentum column is energy-scaled
    torch.testing.assert_close(generator.dp.denorm_edep(_F(sols).edep[:4].double()), e_in.double(), rtol=1e-4, atol=1e-6)
    assert (mask[:4].sum(-1) == 1).all()               # still an interacting event: the E_dep row is generated


def test_without_energy_balance_every_interacting_event_is_flowed(generator: GenerateOut, monkeypatch) -> None:
    set_layout(generator.cond_names)
    rows = _no_outgoing(generator, monkeypatch)
    monkeypatch.setitem(generator.model.rc.odeint_conf, "energy_balance", False)
    generator(_cond(generator, 22, batch=4), passthrough=False)
    assert rows == [4]
