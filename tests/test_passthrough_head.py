"""mm_conf.passthrough_head: a Bernoulli "this particle did not interact" token.

E_dep == 0 and "nothing happened" are the same event in the data (measured
100% coincident), and only neutral primaries ever do it, so the head is masked
to the neutral incoming species.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.multiplicity.model import MultModel


def _config(**mm) -> dict:
    conf = {
        "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
        "mm_conf": {
            "ptypes": torch.tensor([22, 211, 2112], dtype=torch.int64),
            "ptypes_in": torch.tensor([22, 211, 2112, 2212], dtype=torch.int64),
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
            "model_args": {"use_adaptive_rmsnorm": True},
        },
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
    }
    conf["mm_conf"].update(mm)
    return conf


def _model(**mm) -> MultModel:
    torch.manual_seed(0)
    return MultModel({"state_dict": {}, "config": _config(**mm)})


def _inputs(model: MultModel, pdgid_in_idx: int, n: int = 64):
    torch.manual_seed(1)
    return (torch.randn(n, model.rc.in_dim),
            torch.full((n,), pdgid_in_idx, dtype=torch.long))


def test_head_is_absent_by_default() -> None:
    assert _model().pt_head is None


def test_default_allowed_species_are_the_neutrals() -> None:
    m = _model(passthrough_head=True)
    # ptypes_in = [22, 211, 2112, 2212]; only the photon and the neutron
    assert m._pt_allowed.tolist() == [True, False, True, False]


def test_allowed_species_can_be_overridden() -> None:
    m = _model(passthrough_head=True, passthrough_pdgids=[2112])
    assert m._pt_allowed.tolist() == [False, False, True, False]


def test_disabled_head_never_fires() -> None:
    m = _model()
    assert not m.sample_passthrough(*_inputs(m, 0)).any()


def test_charged_incoming_never_fires_however_large_the_logit() -> None:
    m = _model(passthrough_head=True)
    with torch.no_grad():
        m.pt_head[-1].bias.fill_(50.0)
    for charged in (1, 3):  # 211, 2212
        assert not m.sample_passthrough(*_inputs(m, charged)).any()
    for neutral in (0, 2):  # 22, 2112
        assert m.sample_passthrough(*_inputs(m, neutral)).all()


def test_a_fired_event_emits_exactly_the_incoming_particle() -> None:
    m = _model(passthrough_head=True)
    with torch.no_grad():
        m.pt_head[-1].bias.fill_(50.0)
    in_tok, idx = _inputs(m, 2)  # neutron in, ptypes index 2
    skip = m.sample_passthrough(in_tok, idx)
    counts = m((in_tok, None, idx), skip=skip)
    assert counts[:, 2].eq(1).all()
    assert counts[:, [0, 1]].eq(0).all()


def test_skipping_nothing_reproduces_the_plain_decode() -> None:
    """The head must not perturb events that do not fire."""
    m = _model(passthrough_head=True)
    in_tok, idx = _inputs(m, 2)
    torch.manual_seed(5)
    a = m((in_tok, None, idx))
    torch.manual_seed(5)
    b = m((in_tok, None, idx), skip=torch.zeros(len(in_tok), dtype=torch.bool))
    assert torch.equal(a, b)


def test_partial_skip_leaves_the_decoded_rows_untouched() -> None:
    m = _model(passthrough_head=True)
    in_tok, idx = _inputs(m, 2, n=8)
    skip = torch.tensor([True, False] * 4)
    counts = m((in_tok, None, idx), skip=skip)
    assert counts[skip][:, 2].eq(1).all()
    assert counts[skip][:, [0, 1]].eq(0).all()
    assert counts[~skip].shape == (4, 3)


def test_training_step_consumes_the_passthrough_label() -> None:
    m = _model(passthrough_head=True)
    m.trainer = None
    n = 8
    in_tok = torch.randn(n, m.rc.in_dim)
    counts = torch.randint(0, m.rc.max_particles, (n, m.rc.max_seq_len))
    idx = torch.zeros(n, dtype=torch.long)
    lo = m.training_step((in_tok, counts, idx, torch.zeros(n)), 0)
    hi = m.training_step((in_tok, counts, idx, torch.ones(n)), 0)
    assert torch.isfinite(lo) and torch.isfinite(hi)
    assert not torch.allclose(lo, hi)


def test_training_step_without_the_head_ignores_a_fourth_element() -> None:
    m = _model()
    m.trainer = None
    n = 8
    batch = (torch.randn(n, m.rc.in_dim),
             torch.randint(0, m.rc.max_particles, (n, m.rc.max_seq_len)),
             torch.zeros(n, dtype=torch.long))
    assert torch.isfinite(m.training_step(batch, 0))


def test_head_is_nonlinear_in_the_conditioning() -> None:
    """A bare Linear would compose with proj_in into one affine map, which
    cannot bend from 86% pass-through at low density to 24% at high."""
    m = _model(passthrough_head=True)
    x = torch.zeros(7, m.rc.in_dim)
    x[:, 0] = torch.linspace(1.0, 10.0, 7)
    idx = torch.zeros(7, dtype=torch.long)
    with torch.no_grad():
        logit = m.pt_head(m.proj_in(x) + m.embd_pp_(idx)).squeeze(-1)
    slopes = torch.diff(logit) / torch.diff(x[:, 0])
    assert not torch.allclose(slopes, slopes[0], rtol=1e-3)


def test_passthrough_dim_is_configurable() -> None:
    m = _model(passthrough_head=True, passthrough_dim=8)
    assert m.pt_head[0].out_features == 8


def _count_batch(m, n=64, seed=3):
    torch.manual_seed(seed)
    return (torch.randn(n, m.rc.in_dim),
            torch.randint(0, m.rc.max_particles, (n, m.rc.max_seq_len)),
            torch.zeros(n, dtype=torch.long))


def _logits(m, in_tok, counts, idx):
    in_embd = m.proj_in(in_tok) + m.embd_pp_(idx)
    seq = torch.cat((in_embd.unsqueeze(1),
                     m.embd_in_(counts[:, : m.rc.max_seq_len - 1] + m._in_offsets)), dim=1)
    out = m.model(seq, mask=None, condition=in_embd)
    return torch.einsum("bsh,shp->bsp", out, m.proj_out_w) + m.proj_out_b


def test_count_loss_ignores_passthrough_rows() -> None:
    """The decoder must model P(counts | interacted): rewriting the targets of
    the rows the token owns may not move the loss, or their singles stack."""
    m = _model(passthrough_head=True)
    m.trainer = None
    in_tok, counts, idx = _count_batch(m)
    label = torch.zeros(len(in_tok))
    label[::2] = 1.0
    a = m.training_step((in_tok, counts, idx, label), 0)
    masked = counts.clone()
    masked[::2] = (counts[::2] + 1) % m.rc.max_particles
    assert torch.allclose(a, m.training_step((in_tok, masked, idx, label), 0))
    kept = counts.clone()
    kept[1::2] = (counts[1::2] + 1) % m.rc.max_particles
    assert not torch.allclose(a, m.training_step((in_tok, kept, idx, label), 0))


def test_a_passthrough_row_the_token_cannot_own_still_trains_the_decoder() -> None:
    m = _model(passthrough_head=True)
    m.trainer = None
    in_tok, counts, _ = _count_batch(m)
    idx = torch.full((len(in_tok),), 1, dtype=torch.long)  # 211, charged
    label = torch.ones(len(in_tok))
    a = m.training_step((in_tok, counts, idx, label), 0)
    other = (counts + 1) % m.rc.max_particles
    assert not torch.allclose(a, m.training_step((in_tok, other, idx, label), 0))


def test_an_unmasked_batch_is_the_plain_cross_entropy() -> None:
    m = _model()
    m.trainer = None
    in_tok, counts, idx = _count_batch(m)
    expect = torch.nn.functional.cross_entropy(
        _logits(m, in_tok, counts, idx).reshape(-1, m.rc.max_particles), counts.reshape(-1))
    assert torch.allclose(m.training_step((in_tok, counts, idx), 0), expect, atol=1e-6)


def test_validation_rolls_out_with_the_token() -> None:
    m = _model(passthrough_head=True)
    with torch.no_grad():
        m.pt_head[-1].bias.fill_(50.0)
    in_tok, counts, _ = _count_batch(m, n=16)
    idx = torch.full((16,), 2, dtype=torch.long)  # neutron in, ptypes index 2
    m.on_validation_epoch_start()
    m.validation_step((in_tok, counts, idx), 0)
    assert m._val_acc[0].tolist() == [0.0, 0.0, 16.0]
