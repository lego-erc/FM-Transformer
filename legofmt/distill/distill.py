"""Flow-map self-distillation, for few-step sampling.

``one_step_euler_loss`` takes the LightningModule as its first argument. The
target for step ``d`` is two no-grad half-steps at ``d/2``, averaged through
``logmap`` rather than Euclidean-ly because the factors are spheres. Its
bootstrap ladder is anchored at ``d=0`` deliberately -- without that the bottom
rung supervises nothing.
"""

import torch
from torch import Tensor


def one_step_euler_loss(lego, base: Tensor, ds_t, pdgid_idx: Tensor) -> tuple[Tensor, "Tensor | None"]:
    """fp32 wrapper: the loss is ~5e-5 and bf16 autocast inflates it ~390x, so
    the ladder runs outside autocast. Returns ``(kinematic, block)``: the mean over the
    first ``in_dim`` channels, and over the species block or ``None`` without one."""
    with torch.autocast(base.device.type, enabled=False):
        return _one_step_euler_loss_fp32(lego, base.float(), ds_t, pdgid_idx)


def _one_step_euler_loss_fp32(lego, base: Tensor, ds_t, pdgid_idx: Tensor) -> tuple[Tensor, "Tensor | None"]:
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
        x0   = lego.ps.sample(base, ds_t.f.full[..., :-1].float(), t0.squeeze(-1)).x_t  # the flowed state: all but the pdgid
        half = step / 2
        h    = half.unsqueeze(-1)
        d_t  = torch.where(half >= 2.0 ** -(sections - 1), half, torch.zeros_like(half))  # smallest step's teacher uses d=0, anchoring the bootstrap ladder
        s1    = lego.model(x0, t0, d=d_t, **ckw)
        x_mid = torch.where(gen, man.expmap(x0, h * s1), x0)
        s2    = lego.model(x_mid, t0 + half, d=d_t, **ckw)
        x_end = torch.where(gen, man.expmap(x_mid, h * s2), x_mid)
        tgt   = man.logmap(x0, x_end) / step.unsqueeze(-1)
    s_pred = lego.model(x0, t0, d=step, **ckw)
    sq, n, k = (s_pred - tgt) ** 2 * g, g.sum().clamp(min=1), lego.rc.model_args["in_dim"]
    kin = sq[..., :k].sum() / (n * k)
    if s_pred.shape[-1] == k:
        return kin, None
    # the species block: its own mean, weighted by the bare factor (a residual of order r^2 on
    # mis-distilled rows would otherwise lift lv_flow off its floor and cut the kinematic weight)
    return kin, sq[..., k:].sum() / (n * (s_pred.shape[-1] - k))
