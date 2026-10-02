"""base_conf.ang_shape: a second angular parameter on the base distribution.

Without it the tanh-normal theta has one knob (kappa), so the moment loss can
match the mean opening angle or its spread, never both. With it the head emits
an exponent as well, and _base_moment_loss matches the first two moments.
"""

from __future__ import annotations

import math

import pytest
import torch

from legofmt.geometry.geom_trafos import GeomTrafos


def _theta(kappa, ang_pow, n=200000, seed=0):
    torch.manual_seed(seed)
    t = (torch.randn(n) / kappa).tanh().abs()
    return t if ang_pow is None else t.pow(ang_pow)


def test_ang_pow_defaults_to_the_old_sampler() -> None:
    g = GeomTrafos()
    loc = torch.tensor([[[0.0, 0.0, 1.0]]])
    torch.manual_seed(0)
    a = g.sample((1, 8), loc, torch.tensor(8.0), 0.0, True)
    torch.manual_seed(0)
    b = g.sample((1, 8), loc, torch.tensor(8.0), 0.0, True, None)
    torch.manual_seed(0)
    c = g.sample((1, 8), loc, torch.tensor(8.0), 0.0, True, torch.tensor(1.0))
    assert torch.equal(a, b) and torch.allclose(a, c, atol=1e-6)


def test_ang_pow_changes_the_shape_at_a_fixed_mean() -> None:
    """The point of the parameter: two (kappa, pow) pairs with the same mean
    angle but different spread, which one parameter cannot produce."""
    base = _theta(8.0, None)
    m0, s0 = base.mean(), base.std()
    for pw in (0.5, 2.0):
        lo, hi = 0.05, 400.0
        for _ in range(60):
            mid = (lo * hi) ** 0.5
            if _theta(mid, pw).mean() > m0:
                lo = mid
            else:
                hi = mid
        t = _theta((lo * hi) ** 0.5, pw)
        assert abs(t.mean() - m0) / m0 < 0.02
        assert abs(t.std() - s0) / s0 > 0.10


def test_sampler_is_finite_and_on_the_sphere() -> None:
    g = GeomTrafos()
    loc = torch.nn.functional.normalize(torch.randn(4, 1, 3), dim=-1)
    for pw in (None, torch.full((4, 1), 0.4), torch.full((4, 1), 3.0)):
        d = g.sample((4, 6), loc, torch.full((4, 1), 8.0), 0.0, True, pw)
        v = d.view(4, 6, 3)
        assert torch.isfinite(v).all()
        assert torch.allclose(v.norm(dim=-1), torch.ones(4, 6), atol=1e-5)


@pytest.mark.parametrize("ang_shape", [False, True])
def test_head_width_and_defaults(ang_shape) -> None:
    from legofmt.base_dist.base_nn import build_base_head

    class _RC:
        pdgids_template = torch.tensor([11, 22])
        cond_scalars = ("Density", "Z", "A", "Size")
        base_pretrain_batches = 1
        config = {"base_conf": {"ang_shape": ang_shape, "base_head": None}}

    class _GB:
        scale_dist = "sm_norm"
        sm_scale = 0.5
        kappa = 8.0

    head = build_base_head(_RC(), _GB())
    assert head[-1].out_features == (5 if ang_shape else 4)
    if ang_shape:
        assert math.isclose(float(head[-1].bias[4].exp()), 1.0, rel_tol=1e-6)


def test_two_moments_are_matchable_only_with_the_extra_parameter() -> None:
    """Fit (kappa) and (kappa, pow) to a target the one-knob family cannot hold,
    by the same two-moment objective _base_moment_loss uses."""
    torch.manual_seed(0)
    tgt = _theta(3.0, 0.45, seed=7)
    m1_t, m2_t = tgt.mean(), tgt.square().mean()
    z = 2 ** 0.5 * torch.erfinv(torch.linspace(-0.995, 0.995, 400))

    def fit(with_pow):
        lk = torch.tensor(math.log(8.0), requires_grad=True)
        lp = torch.tensor(0.0, requires_grad=True)
        ps = [lk, lp] if with_pow else [lk]
        opt = torch.optim.Adam(ps, lr=0.05)
        for _ in range(3000):
            tb = (z.abs() / lk.exp()).tanh()
            if with_pow:
                tb = tb.pow(lp.exp())
            loss = (tb.mean() - m1_t) ** 2 + (tb.square().mean() - m2_t) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss), float(lk.exp()), float(lp.exp())

    l1, _, _ = fit(False)
    l2, k2, p2 = fit(True)
    assert l2 < l1 / 10
    assert abs(p2 - 0.45) < 0.15 and abs(k2 - 3.0) < 1.0


def _rc(**bc):
    class _RC:
        pdgids_template = torch.tensor([11, 22])
        cond_scalars = ("Density", "Z", "A", "Size")
        base_pretrain_batches = 1
        config = {"base_conf": {"base_head": None, **bc}}
    return _RC()


class _GB:
    scale_dist = "sm_norm"
    sm_scale = 0.5
    kappa = 8.0
    tanh_theta = True
    e_dep_max = 1.0


def test_ang_weight_scales_only_the_angular_term() -> None:
    from legofmt.base_dist.base_nn import BaseDist

    class M(BaseDist):
        def __init__(self, w):
            self.rc = _rc(ang_weight=w)
            self.gen_base = _GB()

    torch.manual_seed(0)
    n, p = 32, 4
    full = torch.randn(n, 2 + 1 + p, 8)
    full[..., 1, 0] = torch.rand(n) * 0.5 + 0.2
    from legofmt.data.struct import DataStruct
    ds = DataStruct(full, torch.ones(n, 3 + p, dtype=torch.long), torch.ones(n, 3 + p, dtype=torch.long))
    fwd = torch.ones(n, dtype=torch.bool)
    s = torch.rand(n, 1) + 0.5
    mu, sig = torch.zeros(n, 1), torch.ones(n, 1)
    kap = torch.full((n, 1), 8.0)
    a = M(1.0)._base_moment_loss(ds, fwd, s, mu, sig, kap)
    b = M(5.0)._base_moment_loss(ds, fwd, s, mu, sig, kap)
    assert b > a


def test_ang_nout_routes_the_angular_columns_through_the_outgoing_pass() -> None:
    from legofmt.base_dist.base_nn import build_base_head

    for nout in (False, True):
        head = build_base_head(_rc(ang_shape=True, base_head_nout=True, ang_nout=nout), _GB())
        assert head[-1].out_features == 5
        assert head[0].in_features == 4 + 2 + (2 + 2)
