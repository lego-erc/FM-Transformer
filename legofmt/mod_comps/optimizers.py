from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def muon_factory(params, **kw):
    from pytorch_optimizer import Muon
    ps = list(params)
    return Muon([
        {"params": [p for p in ps if p.ndim == 2], "use_muon": True},
        {"params": [p for p in ps if p.ndim != 2], "use_muon": False},
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
