"""MultModel.log_prob: log P(counts, pass-through) of a ``forward`` draw, differentiable in the conditioning input --
the score-function term of the density design's gradient (a sampled count cannot be backpropagated through)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from test_helpers import _pt_mult_model as _model


def _batch(m, B: int = 8, seed: int = 5):
    g = torch.Generator().manual_seed(seed)
    S, P = m.rc.max_seq_len, m.rc.max_particles
    return (torch.randn(B, m.rc.in_dim, generator=g), torch.randint(0, P, (B, S), generator=g),
            torch.randint(0, len(m.rc.ptypes_in), (B,), generator=g))


def test_log_prob_equals_minus_summed_ce() -> None:
    m = _model()
    in_tok, counts, idx = _batch(m)
    logits, _, _ = m._teacher_logits(in_tok, counts, idx)
    ce = F.cross_entropy(logits.reshape(-1, m.rc.max_particles), counts.reshape(-1), reduction="none")
    torch.testing.assert_close(m.log_prob(in_tok, counts, idx), -ce.view(counts.shape).sum(-1))


def test_log_prob_matches_sampling_frequencies() -> None:
    m = _model()
    n = 20000
    in_tok = torch.randn(1, m.rc.in_dim, generator=torch.Generator().manual_seed(7)).expand(n, -1)
    idx = torch.zeros(n, dtype=torch.long)
    torch.manual_seed(11)
    with torch.no_grad():
        counts = m((in_tok, None, idx))
    vals, freq = counts.unique(dim=0, return_counts=True)
    top = vals[freq.argmax()]
    p_hat = freq.max().item() / n
    p = m.log_prob(in_tok[:1], top[None], idx[:1]).exp().item()
    assert abs(p_hat - p) <= 4 * (p * (1 - p) / n) ** 0.5, (p_hat, p)


def test_log_prob_gradient_wrt_input() -> None:
    m = _model()
    in_tok, counts, idx = _batch(m)
    in_tok.requires_grad_()
    m.log_prob(in_tok, counts, idx).sum().backward()
    assert torch.isfinite(in_tok.grad).all() and in_tok.grad.abs().sum() > 0


def test_log_prob_passthrough_term() -> None:
    # fired: only the Bernoulli term (the counts are then fixed); not fired, covered species: counts + log(1 - p);
    # species the head does not cover: counts alone
    m = _model(passthrough_head=True)
    in_tok, counts, idx = _batch(m)
    idx = torch.tensor([2, 2, 0, 3, 2, 0, 3, 2])           # 2112 (neutral, covered), 22 (covered), 2212 (charged)
    pt = torch.tensor([True, False, True, False, False, False, False, True])
    lp = m.log_prob(in_tok, counts, idx, pt)
    logits, _, _ = m._teacher_logits(in_tok, counts, idx)
    count_term = logits.log_softmax(-1).gather(-1, counts.unsqueeze(-1)).squeeze(-1).sum(-1)
    p = m.passthrough_prob(in_tok, idx)
    covered = m._pt_pdgid_allowed[idx].bool()
    torch.testing.assert_close(lp[pt], torch.log(p[pt]))
    keep = ~pt & covered
    torch.testing.assert_close(lp[keep], count_term[keep] + torch.log1p(-p[keep]))
    torch.testing.assert_close(lp[~covered], count_term[~covered])


def test_log_prob_refuses_a_symmetry_augmented_model() -> None:
    # sym_aug draws a fresh cube transform in every proj_in call: the counts would be scored under another one
    m = _model()
    m.rc = replace(m.rc, sym_aug=True)
    with pytest.raises(NotImplementedError, match="sym_aug"):
        m.log_prob(*_batch(m))


def test_training_loss_unchanged() -> None:
    # recorded from training_step before it was refactored onto _teacher_logits
    for pt, expected in ((False, 1.3789219856), (True, 2.8463780880)):
        m = _model(passthrough_head=True) if pt else _model()
        m.train()
        in_tok, counts, idx = _batch(m)
        batch = (in_tok, counts, idx) + ((torch.tensor([1, 0] * 4, dtype=torch.bool),) if pt else ())
        assert abs(float(m.training_step(batch, 0)) - expected) < 1e-6
