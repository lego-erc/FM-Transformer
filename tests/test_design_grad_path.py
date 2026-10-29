"""The differentiable generation path: ``_solve`` / ``_forward`` are ``solve`` / ``forward`` without their
``torch.no_grad`` (which stays on the public methods), so a design parameter written into a conditioning row can be
differentiated through a sampled event."""

import pytest
import torch
import torch.nn.functional as F

from legofmt.data.struct import DataStruct
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.main.modules import LEGOLtng

from test_helpers import _fake_batch, _head_batch, _head_config, _tiny_config


def _wake(m: LEGOLtng) -> LEGOLtng:
    """ada-LN-zero layerscale makes every block an identity at init, so no token sees another and a conditioning
    cell cannot reach a generated one; perturb every trainable weight so the field mixes tokens."""
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in m.model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def _velocity_model() -> LEGOLtng:
    cfg = _tiny_config({"step_cond": True})
    cfg["config"]["odeint_conf"] = {"method": "euler", "step_size": 0.25}
    torch.manual_seed(0)
    m = LEGOLtng(cfg)
    m.eval()
    return _wake(m)


def _with_density(ds: DataStruct, dens: torch.Tensor) -> DataStruct:
    f = ds.f.full.clone()
    f[:, 0, 0] = dens
    return DataStruct(f, ds.m.full, ds.am.full)


def test_forward_has_no_graph() -> None:
    m, ds = _velocity_model(), _fake_batch(B=4, edep_generated=True)
    dens = ds.f.full[:, 0, 0].clone().requires_grad_()
    assert m.forward(_with_density(ds, dens))[0].grad_fn is None


def test_private_forward_matches_and_differentiates() -> None:
    m, ds = _velocity_model(), _fake_batch(B=4, edep_generated=True)
    torch.manual_seed(1)
    a = m.forward(ds)[0]
    torch.manual_seed(1)
    b = m._forward(ds)[0]
    torch.testing.assert_close(a, b.detach())
    dens = ds.f.full[:, 0, 0].clone().requires_grad_()
    torch.manual_seed(1)
    out = m._forward(_with_density(ds, dens))[0]
    (g,) = torch.autograd.grad(out[:, 1, 0].sum(), dens)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_head_params_gradient_finite_at_normal_incidence() -> None:
    # the chord divides by each component of the incoming direction; a gun along +z has two zero components, whose
    # discarded 0/0 branch must not reach the backward pass
    m = LEGOLtng(_head_config(True, {"ntypes": 7}))
    ds = _head_batch(torch.full((4,), 0.5), torch.tensor([1, 2, 1, 2]))
    f = ds.f.full.clone()
    f[:, 5, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 5, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f.requires_grad_()   # every cell, the incoming row included: in a sampled event they share one graph
    _, mu, sig, _, _ = m.base_head_params(DataStruct(f, ds.m.full, ds.am.full))
    (g,) = torch.autograd.grad((mu + sig).sum(), f)
    assert torch.isfinite(g).all()


def test_to_sph_gradient_finite_at_the_pole() -> None:
    # a gun along the z axis sits on the pole of the base sampler's spherical coordinates
    cc = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.6, 0.0, 0.8]], requires_grad=True)
    sph = GeomTrafos().to_sph(cc)
    torch.testing.assert_close(sph.detach(), torch.tensor([[0.0, 0.0], [torch.pi, 0.0], [0.6435011, 0.0]]))
    (g,) = torch.autograd.grad(sph.sum(), cc)
    assert torch.isfinite(g).all()


def test_to_sph_gradient_finite_next_to_the_pole() -> None:
    # x^2 + y^2 subnormal (within ~1e-19 of the axis): sqrt's and atan2's backward overflow there, as in rotate
    cc = torch.tensor([[-2.7e-22, -2.5e-22, 1.0]], requires_grad=True)
    sph = GeomTrafos().to_sph(cc)
    assert sph[0, 0].item() < 1e-18
    (g,) = torch.autograd.grad(sph.sum(), cc)
    assert torch.isfinite(g).all()


def test_rotate_gradient_finite_at_the_axis() -> None:
    # a tiny sampled angle rotated by 0 (a gun along z): the acos argument rounds to exactly 1 in fp32
    sph = torch.tensor([[1e-5, 0.3], [0.5, 0.3]], requires_grad=True)
    zero = torch.zeros(2, 1)
    out = GeomTrafos().rotate(sph, zero, zero)
    assert out[0, 0].item() == 0.0 and abs(out[1, 0].item() - 0.5) < 1e-6   # forward as before
    (g,) = torch.autograd.grad(out.sum(), sph)
    assert torch.isfinite(g).all()


@pytest.mark.parametrize("ang_pow", [0.6, None])
def test_sample_gradient_finite_at_a_zero_draw(monkeypatch, ang_pow) -> None:
    # CUDA's normal draw is exactly 0 about once in 2**24 (a chunk of the optimizer draws ~6.5e4): theta = 0 about a gun
    # along z is atan2(0, 0) in rotate, and with the trained head's ang_pow < 1 also t^p's infinite slope at t = 0;
    # either way 0 * inf = NaN reached kappa and the density
    monkeypatch.setattr(torch, "randn", lambda n, device=None: torch.zeros(n, device=device))
    kappa = torch.tensor(3.0, requires_grad=True)
    p = None if ang_pow is None else torch.tensor(ang_pow, requires_grad=True)
    loc = torch.tensor([[0.0, 0.0, 1.0]])
    out = GeomTrafos().sample((4,), loc, kappa, tanh_theta=True, ang_pow=p)
    assert (out == loc.expand(4, -1)).all()   # forward: every direction exactly on the axis
    grads = torch.autograd.grad(out.sum(), [kappa] + ([] if p is None else [p]))
    assert all(torch.isfinite(g) for g in grads)


@pytest.mark.parametrize("theta", [0.0, 3.7e-22])
def test_rotate_gradient_finite_at_the_pole(theta: float) -> None:
    # a sampled angle of 3.7e-22 about the z axis (an optimizer chunk on CUDA): rotate's atan2 backward takes the
    # reciprocal of x^2 + y^2 = 1.4e-43, a subnormal whose reciprocal overflows to inf (exactly 0 torch masks)
    sph = torch.tensor([[theta, 3.895]], requires_grad=True)
    alpha = torch.zeros(1, 1, requires_grad=True)
    out = GeomTrafos().to_cc(GeomTrafos().rotate(sph, alpha, torch.zeros(1, 1)))
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0]]))
    grads = torch.autograd.grad(out.sum() + out[:, 0].sum(), (sph, alpha))
    assert all(torch.isfinite(g).all() for g in grads)


def test_gradient_finite_with_padding() -> None:
    # events with 1 and 2 outgoing particles (a padded slot), whitened E_dep row, zero-velocity conditioning rows
    torch.manual_seed(0)
    m = _wake(LEGOLtng(_head_config(True, {"ntypes": 7}, edep_whiten=True)))
    m.eval()
    ds = _head_batch(torch.full((6,), 0.5), torch.tensor([1, 2, 1, 2, 1, 2]))
    dens = ds.f.full[:, 0, 0].clone().requires_grad_()
    out = m._forward(_with_density(ds, dens))[0]
    (g,) = torch.autograd.grad(out[:, 1, 0].sum() + out[:, 6:, 0].nansum(), dens)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_to_sph_and_rotate_values_are_the_plain_formulas() -> None:
    # the gradient guards only detach inputs at the poles: values stay bit-identical to atan2(hypot(x, y), z),
    # atan2(y, x) and rotate's acos / atan2, so seeded sampling of old checkpoints is unchanged (phi at a pole too)
    g = torch.Generator().manual_seed(0)
    cc = F.normalize(torch.randn(4096, 3, generator=g), dim=-1)
    cc[:3] = torch.tensor([[0.0, 0.0, 1.0], [-0.0, 0.0, 1.0], [3e-20, -2e-20, -1.0]])
    x, y, z = F.normalize(cc, dim=-1).unbind(-1)   # to_sph normalizes its input
    assert torch.equal(GeomTrafos().to_sph(cc), torch.stack((torch.atan2(torch.hypot(x, y), z), torch.atan2(y, x)), -1))
    sph, alpha, beta = GeomTrafos().to_sph(cc), torch.rand(4096, 1, generator=g), torch.rand(4096, 1, generator=g)
    sph[:2, 0] = torch.tensor([1e-5, 3.7e-22])
    alpha[:2] = 0.0
    st, ct, sp, cp = sph[:, :1].sin(), sph[:, :1].cos(), sph[:, 1:].sin(), sph[:, 1:].cos()
    plain = torch.cat((torch.acos((st * sp * alpha.sin() + ct * alpha.cos()).clamp(-1 + 1e-8, 1 - 1e-8)),
                       torch.atan2(st * sp * alpha.cos() - ct * alpha.sin(), st * cp) + beta), -1)
    assert torch.equal(GeomTrafos().rotate(sph, alpha, beta), plain)
