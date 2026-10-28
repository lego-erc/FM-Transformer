"""The sphere factors' expmap must be differentiable at zero velocity: every conditioning row of a solve gets exactly
zero velocity, and flow_matching's ``Sphere.expmap`` hands ``torch.where`` a ``0/0`` there, which turns the whole
backward pass NaN (harmless under no_grad, fatal for differentiating a sampled event)."""

import torch
import torch.nn.functional as F
from flow_matching.utils.manifolds import Sphere

from legofmt.geometry.product_manifold import sphere_expmap
from legofmt.mod_comps.config import build_manifold


def _sphere_points(B: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = F.normalize(torch.randn(B, 3, generator=g), dim=-1)
    u = torch.randn(B, 3, generator=g)
    u = u - (x * u).sum(-1, keepdim=True) * x
    u[:4] = 0.0
    return x, u


def test_expmap_is_flow_matching_sphere_bit_for_bit() -> None:
    x, u = _sphere_points()
    assert torch.equal(sphere_expmap(x, u), Sphere().expmap(x, u))


def test_expmap_gradient_is_finite_at_zero_velocity() -> None:
    x, u = _sphere_points()
    x.requires_grad_()
    u.requires_grad_()
    sphere_expmap(x, u).sum().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(x.grad).all()


def test_product_expmap_gradient_is_finite_on_conditioning_rows() -> None:
    man = build_manifold([{"name": "euclidean", "dim": 1}, {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}])
    g = torch.Generator().manual_seed(1)
    x = torch.cat((torch.rand(8, 1, generator=g), F.normalize(torch.randn(8, 3, generator=g), dim=-1),
                   F.normalize(torch.randn(8, 3, generator=g), dim=-1)), -1)
    u = man.proju(x, torch.randn(8, 7, generator=g))
    u[:3] = 0.0
    x.requires_grad_()
    u.requires_grad_()
    man.expmap(x, u).sum().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(x.grad).all()
