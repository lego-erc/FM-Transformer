"""``method: rk4`` has its own manifold stepper with the species jumps (``_rk4_steps``).

Before, rk4 went through ``flow_matching``'s ``ODESolver`` without the class path: a unified
model emitted only the pinned slots (n_out 0.46 on count_s6). The stepper keeps every stage
instantaneous (``d = 0``), like ``_midpoint_steps``' ``v2``, and projects each stage's velocity
onto ``T_x`` before the geodesic step.
"""

from __future__ import annotations

import torch
from torch import nn

from legofmt.main.modules import LEGOLtng
from test_modules_direct import _fake_batch, _tiny_config


class _Rotation(nn.Module):
    """Velocity ``w x x`` on both sphere factors, zero on the energy channel: the exact flow is
    the rotation by ``|w| t`` about ``w``."""

    def __init__(self, manifold, w) -> None:
        super().__init__()
        self.manifold, self.w = manifold, w

    def forward(self, x, t, d=None, return_species=False, **extras):
        v = torch.zeros_like(x)
        for a in (1, 4):
            v[..., a:a + 3] = torch.cross(self.w.expand_as(x[..., a:a + 3]), x[..., a:a + 3], dim=-1)
        v = v * (extras["mask"] == 1).unsqueeze(-1)
        return (v, torch.zeros(*x.shape[:-1], 4)) if return_species else v


def _rotate(x, w, t):
    th, k = w.norm() * t, w / w.norm()
    out = x.clone()
    for a in (1, 4):
        p = x[..., a:a + 3]
        out[..., a:a + 3] = (p * th.cos() + torch.cross(k.expand_as(p), p, dim=-1) * th.sin()
                             + k * (p * k).sum(-1, keepdim=True) * (1 - th.cos()))
    return out


def test_rk4_is_far_more_accurate_than_euler_on_a_sphere_rotation() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_tiny_config()).eval()
    w = torch.tensor([0.3, -0.4, 1.2])
    m.model = _Rotation(m.model.manifold, w)
    batch = _fake_batch()
    x = batch.f.model_in.clone()
    for a in (1, 4):
        x[..., a:a + 3] = torch.nn.functional.normalize(torch.randn_like(x[..., a:a + 3]), dim=-1)
    gen = (batch.m.full == 1).unsqueeze(-1)
    grid = torch.linspace(0.0, 1.0, 5)
    extras = dict(mask=batch.m.full, attn_mask=batch.am.full, types=m.types_embd,
                  pdgids=torch.zeros_like(batch.m.full))
    sp_gen = torch.zeros_like(batch.m.full, dtype=torch.bool)
    exact = torch.where(gen, _rotate(x, w, torch.tensor(1.0)), x)
    err = {}
    for name, fn in (("euler", m._euler_steps), ("rk4", m._rk4_steps)):
        out, _ = fn(x.clone(), grid, sp_gen, **dict(extras, pdgids=extras["pdgids"].clone()))
        err[name] = (out - exact)[gen.expand_as(out)].abs().max().item()
    assert err["rk4"] < 0.02 * err["euler"], err
    assert err["rk4"] < 1e-3, err


def test_rk4_stages_stay_instantaneous_and_jump_the_species() -> None:
    m = LEGOLtng(_tiny_config()).eval()
    seen: list = []
    m.model.register_forward_pre_hook(lambda _mod, _a, kw: seen.append(kw.get("d", None)), with_kwargs=True)
    _, sp = m.solve(_fake_batch(), method="rk4", step_size=0.5, return_species=True)
    assert len(seen) == 8 and all(d is None for d in seen), seen   # 4 stages x 2 steps, every one at d = 0
    n0 = m.rc.n_prefix + 1
    assert (sp[:, n0:] > 0).any(), "rk4 generated no species"
