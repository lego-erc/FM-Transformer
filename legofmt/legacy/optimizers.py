"""The schedule-free optimizer, superseded by Muon with a warmup-cosine schedule (``mod_comps/optimizers.py``).

Schedule-free AdamW keeps two weight sequences and must be told which one is live: ``opt.train()`` before a training
step, ``opt.eval()`` before any evaluation or checkpoint. ``ScheduleFreeSwitch`` is that switch for both
LightningModules; with any other optimizer every method here does nothing.
"""


def schedulefree_adamw(params, **kw):
    """Constructs ``schedulefree.AdamWScheduleFree`` (``opt_conf.opt: schedulefree``)."""
    from schedulefree import AdamWScheduleFree
    return AdamWScheduleFree(params, **kw)


def default_schedulefree(params, mm_conf: dict):
    """Constructs the count model's optimizer for a config without ``opt_conf``."""
    return schedulefree_adamw(params, lr=mm_conf.get("lr", 1e-3), betas=(0.95, 0.999),
                              weight_decay=mm_conf.get("weight_decay", 0.0),
                              warmup_steps=mm_conf.get("warmup_steps", 0))


def opt_is_schedulefree(opt) -> bool:
    """Returns whether ``opt`` needs the explicit train/eval switch schedule-free carries."""
    return callable(getattr(opt, "train", None))


def opt_train(opt) -> None:
    if opt_is_schedulefree(opt):
        opt.train()


def opt_eval(opt) -> None:
    if opt_is_schedulefree(opt):
        opt.eval()


class ScheduleFreeSwitch:
    """Mixin: puts a schedule-free ``self.opt`` in train / eval mode around Lightning's validation."""

    def _opt_train(self) -> None:
        opt_train(self.opt)

    def _opt_eval(self) -> None:
        opt_eval(self.opt)

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        self._opt_eval()

    def on_validation_model_train(self) -> None:
        super().on_validation_model_train()
        self._opt_train()
