"""Flow-map self-distillation, for few-step sampling.

``one_step_euler_loss`` takes the LightningModule as its first argument. The
target for step ``d`` is two no-grad half-steps at ``d/2``, averaged through
``logmap`` rather than Euclidean-ly because the factors are spheres. Its
bootstrap ladder is anchored at ``d=0`` deliberately -- without that the bottom
rung supervises nothing.

**It runs in fp32 regardless of the trainer's precision.** The target is
``logmap(x0, x_end) / step``: it differences two O(1) states and divides by a
step as small as ``2**-(sections-1)``, which turns bf16's ~2**-8 absolute
rounding into ``2**-8/step`` relative error -- measured 47% at ``d=2**-7``
Euclidean, and 100% on the sphere factors by ``d=2**-5`` because ``arccos`` is
ill-conditioned for near-parallel vectors. Since ``x_end`` comes from two
*no-grad* forwards this is corrupted supervision, not gradient noise, so no
amount of training removes it: under ``bf16-mixed`` the loss plateaus at ~0.027
against ~7e-05 in fp32 (2026-09-14, 600 steps, everything else identical).

Only this function needs the guard. The solvers *accumulate* small increments
into an O(1) state, which keeps ~2**-8 relative error rather than amplifying it
-- bf16 generation measures within 0.98-1.07x of fp32 at every step size.
"""

import torch
from torch import Tensor


def one_step_euler_loss(lego, base: Tensor, ds_t, pdgid_idx: Tensor) -> Tensor:
    with torch.autocast(base.device.type, enabled=False):
        return _one_step_euler_loss_fp32(lego, base.float(), ds_t, pdgid_idx)


def _one_step_euler_loss_fp32(lego, base: Tensor, ds_t, pdgid_idx: Tensor) -> Tensor:
    mask, am = ds_t.m.full, ds_t.am.full
    gen = (mask == 1).unsqueeze(-1)
    g   = gen & am.unsqueeze(-1)
    ckw = dict(mask=mask, attn_mask=am, types=lego.types_embd, pdgids=pdgid_idx)
    man = lego.model.manifold

    sections = lego.rc.one_step_euler_sections
    with torch.no_grad():
        step = 2.0 ** -(torch.randint(1, sections + 1,
                                      (base.shape[0], 1), device=base.device) - 1).to(base.dtype)
        t0   = (torch.rand_like(step) * (1.0 / step).round()).floor() * step  # grid-aligned start
        x0   = lego.ps.sample(base, ds_t.f.model_in.float(), t0.squeeze(-1)).x_t
        half = step / 2
        h    = half.unsqueeze(-1)
        d_t  = torch.where(half >= 2.0 ** -(sections - 1), half, torch.zeros_like(half))
        s1    = lego.model(x0, t0, d=d_t, **ckw)
        x_mid = torch.where(gen, man.expmap(x0, h * s1), x0)
        s2    = lego.model(x_mid, t0 + half, d=d_t, **ckw)
        x_end = torch.where(gen, man.expmap(x_mid, h * s2), x_mid)
        tgt   = man.logmap(x0, x_end) / step.unsqueeze(-1)
    s_pred = lego.model(x0, t0, d=step, **ckw)
    return ((s_pred - tgt) ** 2 * g).sum() / (g.sum().clamp(min=1) * s_pred.shape[-1])
