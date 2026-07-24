import torch
from torch import Tensor


def curvature_loss(lego, x_t: Tensor, t: Tensor, v_out: Tensor, ds_t, pdgid_idx: Tensor) -> Tensor:
    mask, am = ds_t.m.full, ds_t.am.full
    gen = (mask == 1).unsqueeze(-1)
    g   = gen & am.unsqueeze(-1)
    man = lego.model.manifold
    eps = lego.rc.curv_eps
    with torch.no_grad():
        e   = torch.where(t + eps <= 1.0, eps, -eps)  # probe backward near t=1
        x_e = torch.where(
            gen, man.expmap(x_t, e.unsqueeze(-1) * man.proju(x_t, v_out.detach())), x_t)
    v_e = lego.model(x_e, t + e, mask=mask, attn_mask=am, types=lego.types_embd, pdgids=pdgid_idx)
    sq  = man.proju(x_t, v_e - v_out) ** 2            # probe sign cancels: (+-eps)^2 = eps^2
    return (sq * g).sum() / (g.sum().clamp(min=1) * v_out.shape[-1] * eps ** 2)


def one_step_euler_loss(lego, base: Tensor, ds_t, pdgid_idx: Tensor) -> Tensor:
    mask, am = ds_t.m.full, ds_t.am.full
    gen = (mask == 1).unsqueeze(-1)
    g   = gen & am.unsqueeze(-1)
    ckw = dict(mask=mask, attn_mask=am, types=lego.types_embd, pdgids=pdgid_idx)
    man = lego.model.manifold

    with torch.no_grad():
        step = 2.0 ** -(torch.randint(1, lego.rc.one_step_euler_sections + 1,
                                      (base.shape[0], 1), device=base.device) - 1).to(base.dtype)
        t0   = (torch.rand_like(step) * (1.0 / step).round()).floor() * step  # grid-aligned start
        x0   = lego.ps.sample(base, ds_t.f.model_in, t0.squeeze(-1)).x_t
        h    = (step / 2).unsqueeze(-1)
    s_pred = lego.model(x0, t0, **ckw)
    with torch.no_grad():
        x_mid = torch.where(gen, man.expmap(x0, h * s_pred), x0)
        x_end = torch.where(gen, man.expmap(x_mid, h * lego.model(x_mid, t0 + step / 2, **ckw)), x0)
        tgt   = man.logmap(x0, x_end) / step.unsqueeze(-1)
    return ((s_pred - tgt) ** 2 * g).sum() / (g.sum().clamp(min=1) * s_pred.shape[-1])
