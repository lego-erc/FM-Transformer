"""Flow-map self-distillation, for few-step sampling.

``one_step_euler_loss`` takes the LightningModule as its first argument. The
target for step ``d`` is two no-grad half-steps at ``d/2``, averaged through
``logmap`` rather than Euclidean-ly because the factors are spheres. Its
bootstrap ladder is anchored at ``d=0`` deliberately -- without that the bottom
rung supervises nothing.
"""

import torch
from torch import Tensor


def one_step_euler_loss(lego, base: Tensor, ds_t, pdgid_idx: Tensor, model=None, z=None) -> Tensor:
    """fp32 wrapper: the loss is ~5e-5 and bf16 autocast inflates it ~390x, so
    the ladder runs outside autocast."""
    with torch.autocast(base.device.type, enabled=False):
        z = None if z is None else tuple(a.float() for a in z)
        model = lego.model if model is None else model
        return _one_step_euler_loss_fp32(lego, base.float(), ds_t, pdgid_idx, model, z)


def _one_step_euler_loss_fp32(lego, base: Tensor, ds_t, pdgid_idx: Tensor, model, z=None) -> Tensor:
    """``z`` = (base, target) of the ``model_conf.edep_ae`` latent: it takes the same two teacher half-steps
    (Euclidean) and its error joins the mean as K more cells."""
    mask, am = ds_t.m.full, ds_t.am.full
    gen = (mask == 1).unsqueeze(-1)
    g   = gen & am.unsqueeze(-1)
    ckw = dict(mask=mask, attn_mask=am, types=lego.types_embd[:, :mask.shape[1]], pdgids=pdgid_idx)
    man = lego.model.manifold
    field = lambda x, t, zz, **kw: (model(x, t, **kw, **ckw), None) if zz is None else model(x, t, z=zz, **kw, **ckw)

    sections = lego.rc.one_step_euler_sections
    with torch.no_grad():
        step = 2.0 ** -(torch.randint(1, sections + 1, (base.shape[0], 1), device=base.device) - 1).to(base.dtype)
        t0   = (torch.rand_like(step) * (1.0 / step).round()).floor() * step  # grid-aligned start
        x0   = lego.ps.sample(base, ds_t.f.model_in.float(), t0.squeeze(-1)).x_t
        z0   = None if z is None else z[0] + t0 * (z[1] - z[0])
        half = step / 2
        h    = half.unsqueeze(-1)
        # the smallest step's teacher uses d=0, anchoring the bootstrap ladder
        d_t  = torch.where(half >= 2.0 ** -(sections - 1), half, torch.zeros_like(half))
        s1, w1 = field(x0, t0, z0, d=d_t)
        x_mid = torch.where(gen, man.expmap(x0, h * s1), x0)
        z_mid = None if z is None else z0 + half * w1
        s2, w2 = field(x_mid, t0 + half, z_mid, d=d_t)
        x_end = torch.where(gen, man.expmap(x_mid, h * s2), x_mid)
        tgt   = man.logmap(x0, x_end) / step.unsqueeze(-1)
        tgt_z = None if z is None else (z_mid + half * w2 - z0) / step
    s_pred, p_z = field(x0, t0, z0, d=step)
    err, n = ((s_pred - tgt) ** 2 * g).sum(), g.sum() * s_pred.shape[-1]
    if z is not None:
        err, n = err + ((p_z - tgt_z) ** 2).sum(), n + p_z.numel()
    return err / n.clamp(min=1)
