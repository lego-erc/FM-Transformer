"""Flow-map self-distillation, for few-step sampling.

``one_step_euler_loss`` takes the LightningModule as its first argument. The
target for step ``d`` is two no-grad half-steps at ``d/2``, averaged through
``logmap`` rather than Euclidean-ly because the factors are spheres. Its
bootstrap ladder is anchored at ``d=0`` deliberately -- without that the bottom
rung supervises nothing.
"""

import torch
from torch import Tensor


def one_step_euler_loss(lego, base: Tensor, ds_t, pdgid_idx: Tensor) -> Tensor:
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
        x0   = lego.ps.sample(base, ds_t.f.model_in, t0.squeeze(-1)).x_t
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
