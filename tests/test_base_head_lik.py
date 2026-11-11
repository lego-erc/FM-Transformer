"""``base_conf.base_head_lik``: one head produces every base parameter, fitted by likelihood.

Energy ``u = 1 - tanh(sm |z|^q)`` per outgoing slot (out-set zeroed pass); E_dep logit-normal ``(mu, sig)`` and angles
``theta / pi = tanh(|z| / kappa)^pw`` (full pass). Loss: per-unit mean NLLs over continuum slots, outgoing slots,
deposit events.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from legofmt.base_dist.base_nn import BaseHeadLik, _ang_logp, _en_logp
from legofmt.base_dist.gen_base import GenerateBase
from legofmt.data.struct import DataStruct
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.main.modules import LEGOLtng
from test_helpers import _head_batch, _head_config

W0 = 3e-4


def _config(**base_conf) -> dict:
    cfg = _head_config(True)
    cfg["config"]["base_conf"].update({"sm_scale": 0.5, "ang_shape": True, "ang_nout": True, "ang_species": True,
                                       "base_head_lik": True, "base_head_hidden": [8, 8], **base_conf})
    return cfg


def _shaken(model: LEGOLtng, seed: int = 0) -> LEGOLtng:
    """Random readouts too: a fresh head's zero-init readouts would hide any input dependence."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.base_head.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


def _batch(n_out: torch.Tensor, edep: torch.Tensor | None = None) -> DataStruct:
    """``_head_batch`` with the pad slot holding what prep writes there: direction (1, 1, 1)/sqrt3, energy 0."""
    ds = _head_batch(torch.full(n_out.shape, 0.1) if edep is None else edep, n_out)
    pad = n_out == 1
    ds.f.full[pad, 7, 0] = 0.0
    ds.f.full[pad, 7, 1:7] = 3 ** -0.5
    return ds


def _phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-x / 2 ** 0.5)


# ------------------------------------------------------------------ densities
@pytest.mark.parametrize("s, q", [(0.5, 1.0), (1.7, 0.4), (0.2, 2.5)])
def test_energy_density_is_the_derivative_of_the_sampler_cdf(s, q) -> None:
    # P(U <= u) = P(|z| >= (atanh(1 - u) / s)^(1/q)) = 2 (1 - Phi(.)), U = 1 - tanh(s |z|^q)
    cdf = lambda u: 2 * (1 - _phi((torch.atanh(1 - u) / s).pow(1 / q)))
    u = torch.linspace(0.02, 0.98, 49, dtype=torch.float64)
    h = 1e-6
    num = (cdf(u + h) - cdf(u - h)) / (2 * h)
    sv, qv = torch.full_like(u, s), torch.full_like(u, q)
    assert torch.allclose(_en_logp(u, sv, qv, eps=0.0).exp(), num, rtol=1e-5)
    # the outlier mixture: (1 - eps) p + eps
    assert torch.allclose(_en_logp(u, sv, qv).exp(), (1 - 1e-3) * num + 1e-3, rtol=1e-5)


@pytest.mark.parametrize("kap, pw", [(8.0, 1.0), (2.0, 0.5), (0.7, 2.5), (15.0, 0.3)])
def test_angle_density_matches_the_sampler(kap, pw) -> None:
    # P(W <= w) = P(|z| <= kappa atanh(w^(1/pw))) = erf(kappa atanh(w^(1/pw)) / sqrt2), W = theta / pi
    cdf = lambda w: torch.erf(kap * torch.atanh(w.pow(1 / pw)) / 2 ** 0.5)
    w = torch.linspace(0.01, 0.95, 48, dtype=torch.float64)
    h = 1e-7
    k, p = torch.full_like(w, kap), torch.full_like(w, pw)
    assert torch.allclose(_ang_logp(w, k, p, eps=0.0).exp(), (cdf(w + h) - cdf(w - h)) / (2 * h), rtol=1e-5)
    # below W0 (float32 directions do not resolve it) a value is scored as the censored mass P(W < W0)
    w0 = torch.tensor([W0], dtype=torch.float64)
    cen = _ang_logp(torch.tensor([0.0, 1e-4], dtype=torch.float64), k[:2], p[:2])
    assert torch.allclose(cen, ((1 - 1e-3) * cdf(w0) + 1e-3 * W0).log().expand(2), rtol=1e-9)
    # and the sampler draws that law
    torch.manual_seed(0)
    loc = torch.nn.functional.normalize(torch.tensor([[0.3, -0.5, 0.8]]), dim=-1)
    d = GeomTrafos().sample((200000, 1), loc, torch.tensor(kap), 0.0, True, torch.tensor(pw))
    ws = ((d.reshape(-1, 3) * loc).sum(-1).clamp(-1, 1).acos() / torch.pi).double().sort().values
    emp = torch.arange(1, len(ws) + 1, dtype=torch.float64) / len(ws)
    res = ws >= W0
    assert (cdf(ws[res]) - emp[res]).abs().max() < 1.63 / len(ws) ** 0.5


def test_sm_pow_one_draws_bitwise_what_sm_norm_drew() -> None:
    gb = GenerateBase({"base_conf": {"scale_dist": "sm_norm", "sm_scale": 0.7}})
    torch.manual_seed(3)
    got = gb.rd_scale((64, 5), torch.ones(64, 1))
    torch.manual_seed(3)
    assert torch.equal(got, 1 - torch.tanh(torch.randn(64, 5, 1).abs() * 0.7))


def test_rd_scale_draws_with_per_slot_scale_and_shape() -> None:
    gb = GenerateBase({"base_conf": {"scale_dist": "sm_norm"}})
    gb.sm_scale, gb.sm_pow = torch.rand(64, 5, 1) + 0.1, torch.rand(64, 5, 1) * 2 + 0.3
    torch.manual_seed(3)
    got = gb.rd_scale((64, 5), torch.ones(64, 1))
    torch.manual_seed(3)
    assert torch.equal(got, 1 - torch.tanh(torch.randn(64, 5, 1).abs().pow(gb.sm_pow) * gb.sm_scale))


# ------------------------------------------------------------------ construction and loading
@pytest.mark.parametrize("flag", ["tanh_theta", "base_head_nout", "ang_shape", "ang_nout", "ang_species"])
def test_lik_needs_its_layout(flag) -> None:
    with pytest.raises(ValueError, match="base_head_lik"):
        LEGOLtng(_config(**{flag: False}))


def test_lik_is_an_ask_even_without_pretraining_batches() -> None:
    cfg = _config(tanh_theta=False)
    cfg["config"]["model_conf"]["base_pretrain_batches"] = 0
    with pytest.raises(ValueError, match="base_head_lik"):
        LEGOLtng(cfg)
    cfg = _config()
    cfg["config"]["model_conf"]["base_pretrain_batches"] = 0
    with pytest.raises(ValueError, match="base_head_lik"):   # nothing to fit it, nothing to load
        LEGOLtng(cfg)


def test_saved_lik_head_reloads_frozen_and_equal() -> None:
    sd = _shaken(LEGOLtng(_config())).base_head.state_dict()
    sd["mean"], sd["std"] = torch.tensor([0.4, 0.3, 4.0]), torch.tensor([0.2, 0.2, torch.inf])
    model = LEGOLtng(_config(base_head=sd))
    got = model.base_head.state_dict()
    assert got.keys() == sd.keys() and all(torch.equal(got[k], sd[k]) for k in sd)
    assert not any(p.requires_grad for p in model.base_head.parameters())


def test_a_lik_head_that_does_not_load_raises() -> None:
    sd = LEGOLtng(_config(base_head_hidden=[8, 4])).base_head.state_dict()
    with pytest.raises(ValueError, match="does not load"):
        LEGOLtng(_config(base_head=sd))
    seq = LEGOLtng(_config(base_head_lik=False)).base_head.state_dict()   # a moment head's file
    with pytest.raises(ValueError, match="does not load"):
        LEGOLtng(_config(base_head=seq))


def test_a_moment_config_given_a_lik_head_raises() -> None:
    sd = LEGOLtng(_config()).base_head.state_dict()
    with pytest.raises(ValueError, match="base_head_lik"):
        LEGOLtng(_config(base_head_lik=False, base_head=sd))


# ------------------------------------------------------------------ outputs
def test_params_shapes_and_the_moment_head_has_no_energy_shape() -> None:
    n_out = torch.tensor([1, 2, 2])
    model = _shaken(LEGOLtng(_config()))
    n_cls = len(model.pdgids_template) + 1
    sm, mu, sig, kap, pw, q = model.base_head_params(_batch(n_out))
    assert [t.shape for t in (sm, mu, sig, kap, pw, q)] == [(3, n_cls), (3, 1), (3, 1), (3, n_cls), (3, n_cls),
                                                           (3, n_cls)]
    assert (q >= 0.25).all() and (q <= 4.0).all() and (pw >= 0.1).all() and (pw <= 10.0).all()
    out = LEGOLtng(_config(base_head_lik=False)).base_head_params(_batch(n_out))
    assert len(out) == 6 and out[0].shape == (3, 1) and out[5] is None


def test_energy_rows_do_not_see_the_out_set() -> None:
    model = _shaken(LEGOLtng(_config()))
    one, two = (model.base_head_params(_batch(torch.full((4,), n))) for n in (1, 2))
    for k, name in ((0, "sm"), (5, "q")):
        assert torch.equal(one[k], two[k]), f"{name} moved with the outgoing set"
    for k, name in ((1, "mu"), (2, "sig"), (3, "kappa"), (4, "pw")):
        assert not torch.allclose(one[k], two[k]), f"{name} is blind to the outgoing set"


def test_head_runs_at_full_fp32_under_medium_precision() -> None:
    model = _shaken(LEGOLtng(_config()))
    ds = _batch(torch.tensor([1, 2] * 32))
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        ref = model.base_head_params(ds)
        torch.set_float32_matmul_precision("medium")
        got = model.base_head_params(ds)
        assert torch.get_float32_matmul_precision() == "medium", "the head must restore the global setting"
    finally:
        torch.set_float32_matmul_precision(prev)
    assert all(torch.equal(a, b) for a, b in zip(ref, got))


def test_fit_base_head_hands_the_sampler_per_slot_scale_and_shape() -> None:
    model = _shaken(LEGOLtng(_config())).eval()
    ds = _batch(torch.tensor([1, 2, 2]))
    sm, _, _, kap, pw, q = model.base_head_params(ds)
    model.gen_base_wrapper(ds)
    idx = model._species_idx(ds.f.out_p[..., -1], sm.shape[-1])
    gb = model.gen_base
    assert gb.sm_scale.shape == (3, 2, 1) and gb.sm_pow.shape == (3, 2, 1)
    assert torch.equal(gb.sm_scale.squeeze(-1), sm.gather(1, idx))
    assert torch.equal(gb.sm_pow.squeeze(-1), q.gather(1, idx))
    assert torch.equal(gb.kappa, kap.gather(1, idx)) and torch.equal(gb.ang_pow, pw.gather(1, idx))


# ------------------------------------------------------------------ the fit
def _lik_parts(model: LEGOLtng, ds: DataStruct) -> torch.Tensor:
    fwd = ds.m.full[:, model.rc.n_prefix] == 0
    sm, mu, sig, kap, pw, q = model.base_head_params(ds)
    idx = model._species_idx(ds.f.out_p[..., -1], sm.shape[-1])
    s, q, kap, pw = (t.gather(1, idx) for t in (sm, q, kap, pw))
    return model._base_lik_loss(ds, fwd, s, mu, sig, kap, pw, q)


def test_lik_loss_ignores_pad_content() -> None:
    model = _shaken(LEGOLtng(_config()))
    ds = _batch(torch.tensor([1, 2, 1, 2]))
    ref = _lik_parts(model, ds)
    ds.f.full[torch.tensor([0, 2]), 7, :7] = torch.randn(2, 7)
    assert ref.shape == (3,) and torch.isfinite(ref).all()
    assert torch.equal(ref, _lik_parts(model, ds))


def _training_set(n: int, bs: int = 256):
    g = torch.Generator().manual_seed(1)
    for _ in range(n):
        n_out = torch.where(torch.rand(bs, generator=g) < 0.5, torch.tensor(1), torch.tensor(2))
        mu = torch.where(n_out == 1, -2.5, -0.5)
        yield _batch(n_out, torch.sigmoid(mu + 0.3 * torch.randn(bs, generator=g)))


def test_pretraining_lowers_the_lik_loss_and_freezes() -> None:
    model = LEGOLtng(_config())
    held = next(_training_set(1, 1024))
    before = _lik_parts(model, held).sum()
    final = model.pretrain_base(_training_set(60), steps=60)
    after = _lik_parts(model, held).sum()
    assert torch.isfinite(torch.tensor(final)) and after < before - 0.5
    assert not any(p.requires_grad for p in model.base_head.parameters())


def test_lik_pretraining_runs_at_full_fp32_under_medium_precision() -> None:
    # train.py sets "medium" before setup pretrains: the backward must stay IEEE fp32 too
    got = {}
    prev = torch.get_float32_matmul_precision()
    try:
        for prec in ("highest", "medium"):
            torch.manual_seed(0)
            model = LEGOLtng(_config())
            torch.set_float32_matmul_precision(prec)
            model.pretrain_base(_training_set(5), steps=5)
            assert torch.get_float32_matmul_precision() == prec, "pretraining must restore the global setting"
            torch.set_float32_matmul_precision(prev)
            got[prec] = model.base_head.state_dict()
    finally:
        torch.set_float32_matmul_precision(prev)
    assert all(torch.equal(got["highest"][k], got["medium"][k]) for k in got["highest"])


def test_lik_pretraining_needs_steps_and_raises_on_a_nonfinite_loss() -> None:
    with pytest.raises(ValueError, match="steps"):
        LEGOLtng(_config()).pretrain_base(_training_set(2))
    bad = _batch(torch.tensor([2, 2]), torch.tensor([0.1, float("nan")]))
    with pytest.raises(RuntimeError, match="non-finite"):
        LEGOLtng(_config()).pretrain_base(iter([bad]), steps=1)


def test_pretrain_if_needed_standardises_on_the_forward_training_events_first() -> None:
    cfg = _config()
    cfg["config"]["model_conf"].update(base_pretrain_batches=3, base_pretrain_bs=4)
    cfg["config"]["dl_conf"]["num_workers"] = 0
    model = LEGOLtng(cfg)
    ds = _batch(torch.tensor([1, 2] * 6), torch.linspace(0.05, 0.5, 12))
    ds.f.full[:, 5, 0] = torch.linspace(0.2, 0.9, 12)        # incoming energy varies
    ds.f.full[:, 4, 0] = torch.linspace(50.0, 150.0, 12)     # Size varies -> log t varies; the chord does not
    inv = torch.arange(12) % 3 == 0
    ds.f.full[inv, 5, 0] = 5.0                               # inverse-mask events: no loss term, not in the stats
    ds.m.full[inv, model.rc.n_prefix] = 1
    model._train_ds = ds
    model._trainer = SimpleNamespace(is_global_zero=True, strategy=SimpleNamespace(
        root_device=torch.device("cpu"), broadcast=lambda obj, src=0: obj))
    seen = {}
    model.pretrain_base = lambda batches, **kw: seen.update(kw, n=len(list(batches))) or 0.0
    model._pretrain_base_if_needed()
    n_cls = len(model.pdgids_template) + 1
    y = model._base_head_x(ds)[~inv][:, [0, 1 + n_cls, 2 + n_cls]]
    sd = y.std(0)
    assert (sd[[0, 2]] > 1e-3).all() and sd[1] <= 1e-6
    assert torch.allclose(model.base_head.mean, y.mean(0))
    assert torch.allclose(model.base_head.std[[0, 2]], sd[[0, 2]]) and torch.isinf(model.base_head.std[1])
    assert seen["n"] == 3 and seen["steps"] == 3 and seen.get("lr") is None


def test_lik_loss_terms_are_the_per_unit_means() -> None:
    """Energy over continuum slots, both angles (1e-3 outlier) over valid slots, E_dep logit-normal over deposit
    events."""
    model = _shaken(LEGOLtng(_config()))
    n_out = torch.tensor([1, 2, 2, 1, 2, 2])
    edep = torch.tensor([0.1, 0.0, 0.3, 0.2, 0.0, 0.05])       # two zero-deposit events
    ds = _batch(n_out, edep)
    ds.f.full[:, 5, 4:7] = torch.nn.functional.normalize(torch.tensor([-0.8, 0.5, 0.3]), dim=-1)
    ds.f.full[1, 6, 0] = 0.0                                   # two absorbed (atom) slots among the valid ones
    ds.f.full[4, 7, 0] = 0.0
    got = _lik_parts(model, ds)
    sm, mu, sig, kap, pw, q = model.base_head_params(ds)
    idx = model._species_idx(ds.f.out_p[..., -1], sm.shape[-1])
    s, q, kap, pw = (t.gather(1, idx) for t in (sm, q, kap, pw))
    valid, u = ds.am.out_p, ds.f.out_cc[..., 0]
    c = valid & (u > 0)
    assert int(c.sum()) == int(valid.sum()) - 2
    ang = lambda col: (ds.f.out_cc[..., col] * torch.nn.functional.normalize(ds.f.in_cc[..., 0, col], dim=-1)
                       .unsqueeze(1)).sum(-1).clamp(-1, 1).acos() / torch.pi
    wm, wp = ang(slice(1, 4)), ang(slice(4, 7))
    ok = edep > 0
    y = torch.logit(edep[ok].clamp(1e-4, 1 - 1e-4))
    ref = torch.stack((-_en_logp(u, s, q)[c].mean(),
                       -(_ang_logp(wm, kap, pw, eps=1e-3) + _ang_logp(wp, kap, pw, eps=1e-3))[valid].mean(),
                       -torch.distributions.Normal(mu[ok, 0], sig[ok, 0]).log_prob(y).mean().double()))
    assert torch.allclose(got, ref, rtol=1e-6, atol=0)


def test_lik_head_starts_at_the_priors_with_the_k0_fan_in_bound() -> None:
    model = LEGOLtng(_config())
    h, n = model.base_head, len(model.pdgids_template) + 1
    for r in (h.re, h.rd, h.ra):
        assert torch.equal(r.weight, torch.zeros_like(r.weight))
    full = lambda v: torch.full((n,), v)
    assert torch.allclose(h.re.bias, torch.cat((full(torch.tensor(0.5).log().item()), full(0.0))))
    assert torch.equal(h.rd.bias, torch.zeros(2))
    assert torch.allclose(h.ra.bias, torch.cat((full(torch.tensor(8.0).log().item()), full(0.0))))
    bound = (4 + n) ** -0.5                                 # the nonzero block's fan-in: one species' row
    assert 0.9 * bound < h.k0.weight.abs().max() <= bound


def test_params_read_the_head_columns_in_their_documented_order() -> None:
    # [log sm, log q, mu, log sig, log kappa, log pw]: with zero-init readouts every output is its readout's bias
    model = LEGOLtng(_config())
    h, n = model.base_head, len(model.pdgids_template) + 1
    b = torch.linspace(-1, 1, 4 * n + 2)
    with torch.no_grad():
        h.re.bias.copy_(b[:2 * n])
        h.rd.bias.copy_(b[2 * n:2 * n + 2])
        h.ra.bias.copy_(b[2 * n + 2:])
    sm, mu, sig, kap, pw, q = model.base_head_params(_batch(torch.tensor([1, 2])))
    got = torch.cat((sm.log(), q.log(), mu, sig.log(), kap.log(), pw.log()), -1)
    assert torch.allclose(got, b.expand_as(got), atol=1e-6)


def test_energy_and_angle_shapes_are_clamped() -> None:
    model = LEGOLtng(_config())
    n = len(model.pdgids_template) + 1
    ds = _batch(torch.tensor([1, 2]))
    with torch.no_grad():
        for v, sm_lim, q_lim, pw_lim in ((20.0, 8.0, 4.0, 10.0), (-20.0, -8.0, 0.25, 0.1)):
            model.base_head.re.bias.fill_(v)
            model.base_head.ra.bias[n:].fill_(v)
            sm, _, _, _, pw, q = model.base_head_params(ds)
            assert torch.allclose(sm, torch.tensor(sm_lim).exp().expand_as(sm))
            assert torch.equal(q, torch.full_like(q, q_lim)) and torch.equal(pw, torch.full_like(pw, pw_lim))


def test_lik_head_standardises_energy_chord_log_t_and_drops_a_constant_one() -> None:
    torch.manual_seed(0)
    n = 4
    h = BaseHeadLik(n, [8, 8], 0.5, 8.0)
    with torch.no_grad():
        for p in h.parameters():
            p.add_(0.3 * torch.randn(p.shape))
        h.mean.copy_(torch.tensor([0.5, 0.4, 4.0]))
        h.std.copy_(torch.tensor([0.3, 0.2, torch.inf]))   # log t constant on the training set
    x = torch.rand(16, 3 + n + 1 + n)
    x[:, 1:1 + n] = torch.nn.functional.one_hot(torch.arange(16) % n, n).float()
    moved = lambda col: x.clone().index_put_((torch.arange(16), torch.full((16,), col)), torch.full((16,), 9.0))
    assert torch.equal(h(x), h(moved(2 + n))), "the constant log t column must not reach the head"
    assert not torch.allclose(h(x), h(moved(1 + n))), "the chord column must reach the head"


def test_lik_pretraining_lr_schedule_and_clip(monkeypatch) -> None:
    made = []

    class Adam(torch.optim.Adam):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.seen = []
            made.append(self)

        def step(self, *a, **k):
            g = torch.cat([p.grad.flatten() for grp in self.param_groups for p in grp["params"] if p.grad is not None])
            self.seen.append((self.param_groups[0]["lr"], g.norm().item()))
            return super().step(*a, **k)

    steps = 120
    monkeypatch.setattr(torch.optim, "Adam", Adam)
    LEGOLtng(_config()).pretrain_base(_training_set(steps, 64), steps=steps)
    warm = steps // 50
    cos = lambda k: torch.cos(torch.tensor(torch.pi * k / steps)).item()
    want = [3e-3 * min(1.0, (k + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + cos(k))) for k in range(steps)]
    lrs, norms = zip(*made[0].seen)
    assert len(lrs) == steps and lrs == pytest.approx(want, rel=1e-6)
    assert max(norms) <= 1.0 + 1e-5 and max(norms) >= 1.0 - 1e-5, "the grad norm is clipped at 1"


def test_a_moment_head_resets_the_energy_shape() -> None:
    model = LEGOLtng(_config(base_head_lik=False)).eval()
    model.gen_base.sm_pow = torch.full((3, 2, 1), 2.0)        # left by a lik head on the same sampler
    model.gen_base_wrapper(_batch(torch.tensor([1, 2, 2])))
    assert model.gen_base.sm_pow == 1.0


def test_lik_needs_two_hidden_widths() -> None:
    with pytest.raises(ValueError, match="base_head_hidden"):
        LEGOLtng(_config(base_head_hidden=[16]))
