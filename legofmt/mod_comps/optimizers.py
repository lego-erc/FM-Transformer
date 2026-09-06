from collections import defaultdict

import torch
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class BatchedMuon(torch.optim.Optimizer):
    """``pytorch_optimizer.Muon``'s update (Nesterov momentum -> Newton-Schulz
    orthogonalisation for >=2-D weights, AdamW for the rest, decoupled weight
    decay) with the per-parameter Python loop replaced by ``torch._foreach`` ops
    and one Newton-Schulz call per group of equal-shape matrices. Same
    numbers, ~5x fewer kernel launches (flow 18 -> 4 ms, mult 16 -> 3 ms per
    step; both training steps are launch-bound). Equality check:
    tests/test_batched_muon.py."""

    def __init__(
        self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
        weight_decay=0.0, weight_decouple=True, use_adjusted_lr=False,
        adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.0, adamw_eps=1e-10, **_,
    ):
        for g in params:
            if g["use_muon"]:
                g.update(lr=g.get("lr", lr), momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                         weight_decay=g.get("weight_decay", weight_decay), use_adjusted_lr=use_adjusted_lr)
            else:
                g.update(lr=g.get("lr", adamw_lr), betas=adamw_betas, eps=adamw_eps,
                         weight_decay=g.get("weight_decay", adamw_wd))
            g.update(weight_decouple=weight_decouple, step=0)
        super().__init__(params, {})
        self._shape_groups = {}  # id(group) -> [(param indices, lr ratio)], shapes are fixed

    def _groups_by_shape(self, group, ps):
        if id(group) not in self._shape_groups:
            by_shape = defaultdict(list)  # >2-D weights flatten to (dim0, -1), as in pytorch_optimizer
            for i, p in enumerate(ps):
                by_shape[(p.shape[0], p[0].numel())].append(i)
            # pytorch_optimizer's get_adjusted_lr: Moonlight's sqrt(max(1, rows/cols)) if
            # use_adjusted_lr, else the original Muon 0.2*sqrt(max(rows, cols)) -- always applied
            self._shape_groups[id(group)] = [
                (idx, max(1.0, r / c) ** 0.5 if group["use_adjusted_lr"] else 0.2 * max(r, c) ** 0.5)
                for (r, c), idx in by_shape.items()
            ]
        return self._shape_groups[id(group)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            lr, wd = group["lr"], group["weight_decay"]
            group["step"] += 1
            if wd > 0:
                if group["weight_decouple"]:
                    torch._foreach_mul_(ps, 1.0 - lr * wd)
                else:
                    torch._foreach_add_(grads, ps, alpha=wd)
            if group["use_muon"]:
                bufs = [self.state[p].setdefault("momentum_buffer", torch.zeros_like(p)) for p in ps]
                torch._foreach_lerp_(bufs, grads, 1.0 - group["momentum"])
                updates = torch._foreach_lerp(grads, bufs, group["momentum"]) if group["nesterov"] else bufs
                for idx, ratio in self._groups_by_shape(group, ps):
                    stacked = torch.stack([updates[i].view(len(updates[i]), -1) for i in idx])
                    o = zero_power_via_newton_schulz_5(stacked, num_steps=group["ns_steps"])
                    o = o.contiguous().to(ps[idx[0]].dtype)  # one cast for the whole group; unbind/view below are free
                    torch._foreach_add_(
                        [ps[i] for i in idx], [oi.view(ps[i].shape) for i, oi in zip(idx, o.unbind(0))], alpha=-lr * ratio,
                    )
            else:
                b1, b2 = group["betas"]
                st = group["step"]
                exp_avg    = [self.state[p].setdefault("exp_avg", torch.zeros_like(p)) for p in ps]
                exp_avg_sq = [self.state[p].setdefault("exp_avg_sq", torch.zeros_like(p)) for p in ps]
                torch._foreach_lerp_(exp_avg, grads, 1.0 - b1)
                torch._foreach_mul_(exp_avg_sq, b2)
                torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1.0 - b2)
                denom = torch._foreach_sqrt(exp_avg_sq)
                torch._foreach_add_(denom, group["eps"])
                # p -= lr * (m / bc1) / ((sqrt(v) + eps) / sqrt(bc2)), constants folded into `value`
                torch._foreach_addcdiv_(ps, exp_avg, denom, value=-lr * (1.0 - b2 ** st) ** 0.5 / (1.0 - b1 ** st))
        return loss


def muon_factory(params, **kw):
    ps = list(params)
    return BatchedMuon([
        {"params": [p for p in ps if p.ndim >= 2], "use_muon": True},
        {"params": [p for p in ps if p.ndim <  2], "use_muon": False},
    ], **kw)


def schedulefree_adamw(params, **kw):
    from schedulefree import AdamWScheduleFree
    return AdamWScheduleFree(params, **kw)


def warmup_cosine(opt, total_steps, warmup_frac=0.05, eta_min=1e-6):
    n = max(1, int(warmup_frac * total_steps))
    return SequentialLR(opt, milestones=[n], schedulers=[
        LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=n),
        CosineAnnealingLR(opt, T_max=total_steps - n, eta_min=eta_min),
    ])


OPTIMIZERS = {"muon": muon_factory, "schedulefree": schedulefree_adamw}
SCHEDULERS = {"warmup_cosine": warmup_cosine}


def _get(reg, k):
    return reg[k] if isinstance(k, str) else k


def build_optimizer(params, opt_conf):
    """Returns (optimizer, lr_scheduler_dict | None) ready for Lightning."""
    cfg = {**opt_conf}
    sc = cfg.pop("scheduler", None)
    opt = _get(OPTIMIZERS, cfg.pop("opt"))(params, **cfg)
    if sc is None:
        return opt, None
    sc = {**sc}
    cls = _get(SCHEDULERS, sc.pop("cls"))
    interval = sc.pop("interval", "step")
    return opt, {"scheduler": cls(opt, **sc), "interval": interval}


def opt_is_schedulefree(opt) -> bool:
    """Whether ``opt`` needs the explicit train/eval switch schedule-free carries."""
    return callable(getattr(opt, "train", None))


def opt_train(opt) -> None:
    if opt_is_schedulefree(opt):
        opt.train()


def opt_eval(opt) -> None:
    if opt_is_schedulefree(opt):
        opt.eval()
