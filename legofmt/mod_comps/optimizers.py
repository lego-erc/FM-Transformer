"""Optimizer construction, and ``BatchedMuon``.

``BatchedMuon`` reimplements ``pytorch_optimizer.Muon``'s update with
``torch._foreach`` ops and one Newton-Schulz call per group of equal-shape
matrices; both models' steps are kernel-launch-bound, so the library version's
per-matrix Python loop dominated the step. The recipe is ``muon`` with the
``warmup_cosine`` schedule; schedule-free AdamW is :mod:`legofmt.legacy.optimizers`.
"""

from collections import defaultdict

import torch
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from legofmt.legacy.optimizers import schedulefree_adamw


def muon_lr_ratio(rows: int, cols: int, adjusted: bool) -> float:
    """Returns Muon's learning-rate ratio for a weight flattened to ``(rows, cols)``.

    ```
    ratio = sqrt(max(1, rows / cols))      adjusted (use_adjusted_lr): the original Muon's
          = 0.2 sqrt(max(rows, cols))      else: Moonlight's RMS matching
    ```
    """
    return max(1.0, rows / cols) ** 0.5 if adjusted else 0.2 * max(rows, cols) ** 0.5


class BatchedMuon(torch.optim.Optimizer):
    """``pytorch_optimizer.Muon``'s update (Nesterov momentum -> Newton-Schulz
    orthogonalisation for >=2-D weights, AdamW for the rest, decoupled weight
    decay) with the per-parameter Python loop replaced by ``torch._foreach`` ops
    and one Newton-Schulz call per group of equal-shape matrices. Same update up
    to bf16 rounding in the batched Newton-Schulz, ~5x fewer kernel launches
    (flow 18 -> 4 ms, mult 16 -> 3 ms per step; both training steps are
    launch-bound). Closeness check: tests/test_batched_muon.py."""

    def __init__(
        self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5,
        weight_decay=0.0, weight_decouple=True, use_adjusted_lr=False,
        adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.0, adamw_eps=1e-10, **_,
    ):
        muon = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay,
                    use_adjusted_lr=use_adjusted_lr, weight_decouple=weight_decouple)
        adamw = dict(lr=adamw_lr, betas=adamw_betas, eps=adamw_eps, weight_decay=adamw_wd,
                     weight_decouple=weight_decouple)
        for g in params:  # per-group keys win over the constructor defaults, as in the library
            for k, v in (muon if g["use_muon"] else adamw).items():
                g.setdefault(k, v)
            g["step"] = 0
        super().__init__(params, {})
        self._shape_groups = {}  # id(group) -> [(param indices, lr ratio)], shapes are fixed

    @torch.no_grad()
    def step(self, closure=None):
        """Applies one step: the weight decay, then Muon to the >=2-D weights and AdamW to the rest.

        ```
        p *= 1 - lr wd      weight_decouple; else g += wd p
        ```
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group, ps, grads in self._groups_with_grads():
            lr, wd = group["lr"], group["weight_decay"]
            group["step"] += 1
            if wd > 0 and group["weight_decouple"]:
                torch._foreach_mul_(ps, 1.0 - lr * wd)
            elif wd > 0:
                torch._foreach_add_(grads, ps, alpha=wd)
            if group["use_muon"]:
                self._muon_update(group, ps, grads, lr)
            else:
                self._adamw_update(group, ps, grads, lr)
        return loss

    def _muon_update(self, group, ps, grads, lr) -> None:
        """Updates >=2-D weights with Muon.

        ```
        buf  = mu buf + (1 - mu) g
        upd  = mu buf + (1 - mu) g             nesterov; else upd = buf
        p   -= lr ratio NS(upd)                NS: Newton-Schulz orthogonalisation of upd as a (rows, cols) matrix
        ```

        ``mu`` is ``momentum``, ``ratio`` is ``muon_lr_ratio``. One NS call takes each group of equal-shape matrices.
        """
        mu   = group["momentum"]
        bufs = self._state(ps, "momentum_buffer")
        torch._foreach_lerp_(bufs, grads, 1.0 - mu)                                  # buf
        upd  = torch._foreach_lerp(grads, bufs, mu) if group["nesterov"] else bufs    # upd
        for idx, ratio in self._groups_by_shape(group, ps):
            ns = _newton_schulz(upd, idx, ps, group["ns_steps"])
            torch._foreach_add_([ps[i] for i in idx], ns, alpha=-lr * ratio)            # p

    def _adamw_update(self, group, ps, grads, lr) -> None:
        """Updates 1-D params (gates, biases, scales) with plain AdamW.

        ```
        m   = b1 m + (1 - b1) g
        v   = b2 v + (1 - b2) g^2
        p  -= lr sqrt(1 - b2^t) / (1 - b1^t) m / (sqrt(v) + eps)
        ```
        """
        b1, b2 = group["betas"]
        t = group["step"]
        m = self._state(ps, "exp_avg")
        v = self._state(ps, "exp_avg_sq")
        torch._foreach_lerp_(m, grads, 1.0 - b1)                   # m
        torch._foreach_mul_(v, b2)                                  # v
        torch._foreach_addcmul_(v, grads, grads, value=1.0 - b2)
        denom = torch._foreach_sqrt(v)                              # sqrt(v) + eps
        torch._foreach_add_(denom, group["eps"])
        torch._foreach_addcdiv_(ps, m, denom, value=-lr * (1.0 - b2 ** t) ** 0.5 / (1.0 - b1 ** t))   # p

    # ---- plumbing ------------------------------------------------------------------------------------------------

    def _groups_with_grads(self):
        """Yields ``(group, ps, grads)`` of every group with a gradient, ``ps`` its parameters that have one."""
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            if ps:
                yield group, ps, [p.grad for p in ps]

    def _state(self, ps, name):
        """Returns the state ``name`` of every parameter, zeros at the first step."""
        return [self.state[p].setdefault(name, torch.zeros_like(p)) for p in ps]

    def _groups_by_shape(self, group, ps):
        """Returns the groups of equal-shape matrices among ``ps`` (indices), each with its ``muon_lr_ratio``.

        ``(rows, cols)`` is a weight flattened to ``(dim0, -1)``, as pytorch_optimizer's ``get_adjusted_lr`` does.
        """
        # keyed on the live parameter set, not the group: the index is into `ps`,
        # which drops parameters without gradients, so a freeze after the first
        # step (a pretrained head) would otherwise index past the end
        key = (id(group), tuple(id(p) for p in ps))
        if key not in self._shape_groups:
            by_shape = defaultdict(list)  # >2-D weights flatten to (dim0, -1), as in pytorch_optimizer
            for i, p in enumerate(ps):
                by_shape[(p.shape[0], p[0].numel())].append(i)
            self._shape_groups[key] = [
                (idx, muon_lr_ratio(r, c, group["use_adjusted_lr"])) for (r, c), idx in by_shape.items()
            ]
        return self._shape_groups[key]


def muon_factory(params, **kw):
    """Constructs Muon for ``ndim >= 2`` weights and AdamW for the rest.

    1-D gates and log-variances must not be orthogonalised, which would discard their magnitude.
    """
    ps = list(params)
    return BatchedMuon([
        {"params": [p for p in ps if p.dim() >= 2], "use_muon": True},
        {"params": [p for p in ps if p.dim() <  2], "use_muon": False},
    ], **kw)


def warmup_cosine(opt, total_steps, warmup_frac=0.05, eta_min=1e-6):
    """Constructs a linear warmup over ``warmup_frac``, then a cosine to ``eta_min``.

    ``total_steps`` must be an upper bound on the real step count: past ``T_max`` the cosine rises back toward the peak
    LR.
    """
    n = max(1, int(warmup_frac * total_steps))
    return SequentialLR(opt, milestones=[n], schedulers=[
        LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=n),
        CosineAnnealingLR(opt, T_max=total_steps - n, eta_min=eta_min),
    ])


OPTIMIZERS = {"muon": muon_factory, "schedulefree": schedulefree_adamw}   # schedulefree: legacy
SCHEDULERS = {"warmup_cosine": warmup_cosine}


def build_optimizer(params, opt_conf):
    """Returns (optimizer, lr_scheduler_dict | None) ready for Lightning."""
    cfg = {**opt_conf}
    sc = cfg.pop("scheduler", None)
    opt = _get(OPTIMIZERS, cfg.pop("opt"))(params, **cfg)
    if sc is None:
        return opt, None
    sc = {**sc}
    interval = sc.pop("interval", "step")
    return opt, {"scheduler": _get(SCHEDULERS, sc.pop("cls"))(opt, **sc), "interval": interval}


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _get(reg, k):
    return reg[k] if isinstance(k, str) else k


def _newton_schulz(upd, idx, ps, steps):
    """Returns NS(upd[i]) of the equal-shape parameters in ``idx`` from one batched call, in their shape and dtype."""
    stacked = torch.stack([upd[i].flatten(1) for i in idx])
    o = zero_power_via_newton_schulz_5(stacked, num_steps=steps)
    o = o.contiguous().to(ps[idx[0]].dtype)  # one cast for the whole group; unbind/view below are free
    return [oi.reshape(ps[i].shape) for i, oi in zip(idx, o.unbind(0))]
