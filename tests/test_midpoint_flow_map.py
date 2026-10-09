"""Midpoint gives its half-step stage the flow map, and keeps v2 instantaneous.

``_euler_steps`` passes ``d = dt`` unconditionally; ``_midpoint_steps`` now does
the same for ``v1``, whose only role is to advance ``dt/2``. ``v2`` deliberately
keeps ``d = 0``: it is evaluated at the midpoint but applied as a full step from
``x``, which is where the scheme's second-order accuracy comes from. Handing it
``d = dt`` asks for the average velocity over ``[t+dt/2, t+3dt/2]`` -- a window
shifted by half a step -- and measured ~3x worse MMD at 4 and 8 NFE.

Measured on ``flow/kin_1step_040926`` (rho=3, N=16383, 5 seeds, paired):
4 NFE MMD 0.01850 -> 0.00822, a 22-sigma separation. At 8 NFE the gain is not
significant (-1.1 sigma), consistent with ``emb_d`` vanishing as ``d -> 0``.

No flag, matching ``_euler_steps``: a checkpoint that never trained the flow
map has ``step_gain`` at its zero init, so ``d`` cannot change its output. That
is asserted here, because it is the property that makes the change safe.
"""

from __future__ import annotations

import torch

from legofmt.main.modules import LEGOLtng
from test_modules_direct import _fake_batch, _tiny_config


def _model(step_cond: bool = True) -> LEGOLtng:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["model_args"]["step_cond"] = step_cond
    cfg["config"]["model_conf"]["base_pretrain_batches"] = 0
    return LEGOLtng(cfg)


def _record_d(m: LEGOLtng) -> list:
    seen: list = []
    m.model.register_forward_pre_hook(
        lambda _mod, _a, kw: seen.append(kw.get("d", None)), with_kwargs=True,
    )
    return seen


def test_v1_gets_half_the_step_and_v2_stays_instantaneous() -> None:
    m = _model()
    seen = _record_d(m)
    m.eval()
    m.rc.odeint_conf.update({"method": "midpoint", "step_size": 0.5})
    m(_fake_batch())

    assert len(seen) >= 2, seen
    v1_d, v2_d = seen[0], seen[1]
    assert isinstance(v1_d, torch.Tensor), f"v1 got d={v1_d!r}, want a tensor"
    assert torch.allclose(v1_d.float(), torch.tensor(0.25)), f"v1 d={v1_d}"
    assert v2_d is None, f"v2 must stay instantaneous, got d={v2_d!r}"


def test_euler_still_passes_the_whole_step() -> None:
    m = _model()
    seen = _record_d(m)
    m.eval()
    m.rc.odeint_conf.update({"method": "euler", "step_size": 0.5})
    m(_fake_batch())

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
