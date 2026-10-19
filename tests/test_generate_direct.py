"""Smoke test for :class:`legofmt.distill.reflow.GenerateOutDirect`.

Builds a tiny flow checkpoint on disk (no HF download), instantiates
``GenerateOut`` from it, and runs one end-to-end forward on a synthetic
``cond`` tensor.
"""

from __future__ import annotations

import pytest
import torch

from test_helpers import _save_flow_ckpt, _save_mult_ckpt
from legofmt.data.struct import _F
from legofmt.distill.reflow import GenerateOutDirect as GenerateOut
from legofmt.main.generate import GenerateIn
from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("direct_ckpts")
    return GenerateOut(str(_save_flow_ckpt(tmp)), str(_save_mult_ckpt(tmp)), device="cpu")


def _dummy_cond(gen: GenerateOut, batch: int = 2) -> torch.Tensor:
    """``[B, 8]`` cond: density, E-scaled mom (3), entry pos (3), pdgid."""
    cond = torch.zeros(batch, 8)
    cond[:, 0] = 1.0
    cond[:, 1:4] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, 7] = gen.pdgids[0].item()
    return cond


def test_instantiation(generator: GenerateOut) -> None:
    assert isinstance(generator.model, torch.nn.Module)
    assert generator.pdgids.numel() > 0
    assert generator.max_seq_l > 3


@torch.no_grad()
def test_forward_shapes(generator: GenerateOut) -> None:
    sols, mask, attn_mask = generator(_dummy_cond(generator, batch=2))
    L = generator.max_seq_l
    assert sols.shape == (2, L, 8)
    assert mask.shape == attn_mask.shape == (2, L)
    assert torch.isfinite(sols[attn_mask]).all(), "non-finite values at attended slots"


@torch.no_grad()
def test_unknown_incoming_pdgid_raises(generator: GenerateOut) -> None:
    """An incoming species outside the mult model's ptypes_in used to be relabelled
    as its searchsorted neighbour, silently generating the wrong particle's shower."""
    cond = _dummy_cond(generator)
    cond[0, -1] = 2112.0  # neighbour of 2212 in ptypes_in = (22, 211, 2212)
    with pytest.raises(ValueError, match="ptypes_in"):
        generator(cond)


@torch.no_grad()
def test_saturation_refill_adds_no_unpredicted_species(generator: GenerateOut, monkeypatch) -> None:
    """Counts (22: 5, 211: 1) exceed the 4 outgoing slots; flooring the rescaled counts
    leaves one slot, which used to be drawn with weight 1 for every species -- 2212,
    which the multiplicity model never predicted, included."""
    B = 2000
    monkeypatch.setattr(type(generator.gen_mult), "forward",
                        lambda self, batch, pt_mask=None: torch.tensor([[5, 1]]).expand(B, -1))
    cond = generator._model_cond(_dummy_cond(generator, batch=B), prepped=False)
    cond_fm, _, attn = generator.gen_batch(cond, passthrough=False)
    out_idx = _F(cond_fm).out_p[..., -1]
    assert (attn[:, generator.n_prefix + 1:].sum(-1) == 4).all()
    assert not (out_idx == 3).any(), "a 2212 (flow index 3) was refilled"


def _fake_shower(B: int = 2, L: int = 7) -> tuple:
    """Synthetic shower with RAW PDG ids in the last column, as a dataset
    batch would carry them."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 1] = 0.1
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
    """Inverse generation from a raw-PDG-id shower batch (offline)."""
    tmp = tmp_path_factory.mktemp("in_ckpts")
    gen_in = GenerateIn(
        str(_save_flow_ckpt(tmp, LEGOLtngVelocity)), str(_save_mult_ckpt(tmp)), device="cpu",
    )
    in_p = gen_in(_fake_shower())
    assert in_p.shape == (2, 1, 8)
    assert torch.isfinite(in_p).all(), "non-finite incoming features"
    raw_ids = in_p[..., -1].flatten().long()
    assert all(i in gen_in.pdgids for i in raw_ids), raw_ids
