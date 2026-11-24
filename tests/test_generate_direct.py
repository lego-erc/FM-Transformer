"""Offline tests of :class:`legofmt.legacy.reflow.GenerateOutDirect` and ``GenerateIn``, end to end on tiny checkpoints
written to disk (no HF download) and a synthetic ``cond``."""

from __future__ import annotations

import pytest
import torch

from test_helpers import _gun_cond, _save_flow_ckpt, _save_mult_ckpt
from legofmt.data.struct import _F
from legofmt.legacy.reflow import GenerateOutDirect as GenerateOut
from legofmt.main.generate import GenerateIn
from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("direct_ckpts")
    return GenerateOut(str(_save_flow_ckpt(tmp)), str(_save_mult_ckpt(tmp)), device="cpu")


def test_instantiation(generator: GenerateOut) -> None:
    assert isinstance(generator.model, torch.nn.Module)
    assert generator.pdgids.numel() > 0
    assert generator.max_seq_l > 3


@torch.no_grad()
def test_forward_shapes(generator: GenerateOut) -> None:
    sols, mask, attn_mask = generator(_gun_cond(generator.pdgids[0].item(), batch=2))
    L = generator.max_seq_l
    assert sols.shape == (2, L, 8)
    assert mask.shape == attn_mask.shape == (2, L)
    assert torch.isfinite(sols[attn_mask]).all(), "non-finite values at attended slots"


@torch.no_grad()
def test_unknown_incoming_pdgid_raises(generator: GenerateOut) -> None:
    """An incoming species outside the multiplicity model's ``ptypes_in`` must raise: relabelled as its
    ``searchsorted`` neighbour, it would silently generate another species' event."""
    cond = _gun_cond(generator.pdgids[0].item(), batch=2)
    cond[0, -1] = 2112.0  # neighbour of 2212 in ptypes_in = (22, 211, 2212)
    with pytest.raises(ValueError, match="ptypes_in"):
        generator(cond)


@torch.no_grad()
def test_saturation_refill_adds_no_unpredicted_species(generator: GenerateOut, monkeypatch) -> None:
    """Counts (22: 5, 211: 1) exceed the 4 outgoing slots, and flooring the rescaled counts (3, 0) frees one slot.
    Its species must be sampled in proportion to the counts: never 2212, which the multiplicity model did not
    predict."""
    B = 2000
    monkeypatch.setattr(type(generator.gen_mult), "forward",
                        lambda self, batch, pt_mask=None, rng=None: torch.tensor([[5, 1]]).expand(B, -1))
    cond = generator._model_cond(_gun_cond(generator.pdgids[0].item(), batch=B), prepped=False)
    cond_fm, _, attn = generator.gen_batch(cond, passthrough=False)
    out_idx = _F(cond_fm).out_p[..., -1]
    assert (attn[:, generator.n_prefix + 1:].sum(-1) == 4).all()
    assert not (out_idx == 3).any(), "a 2212 (flow index 3) was refilled"


def _counts_of(out_pid: torch.Tensor) -> list:
    """Returns each event's outgoing count per species of the multiplicity model's ``ptypes`` = (22, 211)."""
    return torch.stack([(out_pid == p).sum(1) for p in (22, 211)], dim=1).tolist()


@torch.no_grad()
def test_gt_mult_replaces_the_sampled_counts(generator: GenerateOut, monkeypatch) -> None:
    """``gt_mult`` takes the place of the multiplicity model's sample, so that the flow is evaluated on its own: the
    multiplicity model is not run, the slots hold exactly the given counts, and no event is a pass-through."""
    def sampled(*args, **kwargs):
        raise AssertionError("the multiplicity model was sampled")
    monkeypatch.setattr(type(generator.gen_mult), "forward", sampled)
    monkeypatch.setattr(type(generator.gen_mult), "sample_passthrough", sampled)
    gt_mult = torch.tensor([[2, 1], [0, 3], [1, 0]])

    sols, mask, _ = generator(_gun_cond(generator.pdgids[0].item(), batch=3), gt_mult=gt_mult)

    assert _counts_of(generator.pdgids_of(_F(sols).out_p[..., -1])) == gt_mult.tolist()
    assert (mask.sum(-1) > 0).all(), "an event generates nothing: it was treated as a pass-through"


@torch.no_grad()
def test_gt_mult_through_the_g4_arguments(generator: GenerateOut) -> None:
    """``gen_model_w_g4_args`` takes ``gt_mult`` per generated event: ``B * n`` rows, in the events' order."""
    gt_mult = torch.tensor([[1, 1], [0, 2], [3, 0], [2, 2]])   # 2 energies, n = 2 events each

    out = generator.gen_model_w_g4_args(
        2, torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([150.0, 300.0]),
        torch.tensor([1.0]), torch.tensor([1.0]), generator.pdgids[:1], gt_mult=gt_mult)

    assert _counts_of(generator.pdgids_of(out["per_particle"]["Outgoing"][..., -1])) == gt_mult.tolist()


@torch.no_grad()
@pytest.mark.parametrize("shape", [(3, 3), (2, 2)])   # a species too many; an event too few
def test_gt_mult_of_another_shape_raises(generator: GenerateOut, shape: tuple) -> None:
    """``gt_mult`` is one row per event and one column per species of ``ptypes``. A column too many would be
    read as if it were not there, and the events would be generated from the wrong counts."""
    with pytest.raises(ValueError, match="gt_mult"):
        generator(_gun_cond(generator.pdgids[0].item(), batch=3), gt_mult=torch.ones(shape, dtype=torch.long))


def _fake_shower(B: int = 2, L: int = 7) -> tuple:
    """Returns a synthetic batch ``(f, m, am)`` with raw PDG ids in the pdgid channel, as a dataset batch has them."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 3:, 0] = 0.5
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 1:4] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 4:7] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 7] = torch.tensor([22.0, 211.0, 22.0, 211.0])
    m = torch.zeros(B, L, dtype=torch.long)
    m[:, 3:] = 1
    am = torch.ones(B, L, dtype=torch.bool)
    return f, m, am


@torch.no_grad()
def test_generate_in_smoke(tmp_path_factory: pytest.TempPathFactory) -> None:
    """``GenerateIn`` infers the incoming particle from a batch with raw PDG ids, on tiny offline checkpoints."""
    tmp = tmp_path_factory.mktemp("in_ckpts")
    gen_in = GenerateIn(
        str(_save_flow_ckpt(tmp, LEGOLtngVelocity)), str(_save_mult_ckpt(tmp)), device="cpu",
    )
    in_p = gen_in(_fake_shower())
    assert in_p.shape == (2, 1, 8)
    assert torch.isfinite(in_p).all(), "non-finite incoming features"
    raw_ids = in_p[..., -1].flatten().long()
    assert all(i in gen_in.pdgids for i in raw_ids), raw_ids


def test_pdgids_of_is_exact_for_nuclei(generator: GenerateOut, monkeypatch) -> None:
    """The output's species indices map to exact int64 PDG ids: a float32 channel would round a nucleus' 10 digits."""
    monkeypatch.setattr(generator, "pdgids", torch.tensor([22, 1000010020, 1000010030]))
    assert generator.pdgids_of(torch.tensor([[0.0, 2.0, 3.0]])).tolist() == [[0, 1000010020, 1000010030]]
