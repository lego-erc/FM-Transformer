"""Midpoint gives its half-step stage the flow map and keeps v2 instantaneous.

Like the Euler steps (``d = dt``), the midpoint step passes ``v1``, which only advances ``dt/2``, ``d = dt/2``
(``flow/kin_1step_040926``, rho=3, N=16383, 5 paired seeds: 4 NFE MMD 0.01850 -> 0.00822, 22 sigma; at 8 NFE
-1.1 sigma, as ``emb_d`` vanishes for ``d -> 0``). ``v2`` keeps ``d = 0``: it is evaluated at the midpoint but applied
as a full step from ``x``, the source of the scheme's second order; ``d = dt`` would ask for the average velocity
over a window shifted by half a step (~3x worse MMD at 4 and 8 NFE). No flag is needed: a checkpoint that never
trained the flow map has ``step_gain`` at its zero init, so ``d`` cannot change its output -- asserted below.
"""

from __future__ import annotations

import torch

from legofmt.main.modules import LEGOLtng
from test_helpers import _fake_batch, _tiny_config


def _model() -> LEGOLtng:
    return LEGOLtng(_tiny_config({"step_cond": True}))


def _stage_d(method: str) -> list:
    """The ``d`` each field evaluation of a step-0.5 solve receives."""
    m = _model()
    seen: list = []
    m.model.register_forward_pre_hook(lambda _mod, _a, kw: seen.append(kw.get("d", None)), with_kwargs=True)
    m.eval()
    m.rc.odeint_conf.update({"method": method, "step_size": 0.5})
    m(_fake_batch())
    return seen


def test_v1_gets_half_the_step_and_v2_stays_instantaneous() -> None:
    seen = _stage_d("midpoint")
    assert len(seen) >= 2, seen
    v1_d, v2_d = seen[0], seen[1]
    assert isinstance(v1_d, torch.Tensor), f"v1 got d={v1_d!r}, want a tensor"
    assert torch.allclose(v1_d.float(), torch.tensor(0.25)), f"v1 d={v1_d}"
    assert float(v2_d) == 0.0, f"v2 must stay instantaneous, got d={v2_d!r}"


def test_euler_still_passes_the_whole_step() -> None:
    seen = _stage_d("euler")
    assert seen and isinstance(seen[0], torch.Tensor), seen
    assert torch.allclose(seen[0].float(), torch.tensor(0.5)), seen[0]


def test_zero_step_gain_makes_d_inert() -> None:
    """Why no flag is needed: without a trained flow map, `d` cannot move it."""
    m = _model()
    assert float(m.model.vf.step_gain.abs().max()) == 0.0, "step_gain must init to 0"

    batch = _fake_batch()
    x = batch.f.model_in
    t = torch.full((x.shape[0], 1), 0.3)
    kw = dict(mask=batch.m.full, attn_mask=batch.am.full,
              types=m.types_embd, pdgids=torch.zeros_like(batch.m.full))
    m.model.eval()
    with torch.no_grad():
        a = m.model(x, t, **kw, d=None)
        b = m.model(x, t, **kw, d=torch.full_like(t, 0.25))
    assert torch.equal(a, b), (a - b).abs().max()
