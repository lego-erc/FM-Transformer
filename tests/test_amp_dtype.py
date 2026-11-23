"""The autocast dtype of a solve: the checkpoint's precision stamp, which ``odeint_conf["amp"]`` overrides.

``run.precision`` is the autocast of training. ``scripts/train.py`` stamps it into the checkpoint as
``additional.precision``, and ``config.amp_dtype`` maps the stamp to the autocast dtype of a solve. A checkpoint
without a stamp resolves to ``None`` and is solved in fp32. Solving a bf16-trained checkpoint under bf16 is 1.6x
faster than in fp32 (2-step midpoint, N=16383: 0.1426 -> 0.0898 ms/event, MMD vs Geant4 0.0321 -> 0.0286).

``odeint_conf["amp"]`` overrides the stamp: absent means "use the stamp", "32" or ``None`` forces fp32. The override
exists because a solve under bf16 is as good however the checkpoint was trained (80 NFE, rho=3, N=16383: MMD of an
fp32-trained checkpoint 0.0100 in fp32 / 0.0097 under bf16, of a bf16-trained one 0.0108 / 0.0111).
``compat.fp32_attention`` keeps the attention of checkpoints from before x-transformers 2.25.5 (logits ``1000*cos``)
in fp32; that is not tested here.
"""

from __future__ import annotations

import torch

from legofmt.main.modules import LEGOLtng
from legofmt.config import amp_dtype, resolve_legoltng_config
from test_helpers import _fake_batch, _tiny_config

BF16_STAMP = {"precision": "bf16-mixed, medium"}


def test_precision_specs_map_to_autocast_dtypes() -> None:
    for want, specs in ((torch.bfloat16, ("bf16-mixed", "bf16", "bf16-true", "bf16-mixed, medium", torch.bfloat16)),
                        (torch.float16, ("16-mixed", "16", "16-true", "16-mixed, high", torch.float16)),
                        (None, ("32", "32, medium", "64", "transformer-engine", "", None, 32, torch.float32))):
        for s in specs:
            assert amp_dtype(s) is want, repr(s)


def test_resolved_config_carries_amp_dtype() -> None:
    assert resolve_legoltng_config(_tiny_config()).amp_dtype is None, "no stamp => fp32"
    cfg = _tiny_config()
    cfg["config"]["additional"] = BF16_STAMP
    assert resolve_legoltng_config(cfg).amp_dtype is torch.bfloat16


def _model(stamp: dict | None = None) -> LEGOLtng:
    cfg = _tiny_config()
    if stamp:
        cfg["config"]["additional"] = dict(stamp)
    return LEGOLtng(cfg)


def _dtypes_seen(model, **amp) -> list[torch.dtype]:
    """Runs one solve with ``amp`` merged into ``odeint_conf``; returns the autocast dtype of each call of the field."""
    model.rc.odeint_conf.update(method="euler", step_size=1.0, **amp)
    seen: list[torch.dtype] = []
    hook = model.model.register_forward_pre_hook(lambda *_a, **_k: seen.append(
        torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else torch.float32))
    try:
        model.eval()
        model(_fake_batch())
    finally:
        hook.remove()
    assert seen, "the vector field was never called"
    return seen


def test_absent_key_uses_the_stamp() -> None:
    m = _model(BF16_STAMP)
    assert m.rc.amp_dtype is torch.bfloat16
    assert "amp" not in m.rc.odeint_conf, "fixture must not preset the key"
    assert _dtypes_seen(m) == [torch.bfloat16]


def test_override_forces_bf16_on_an_fp32_checkpoint() -> None:
    m = _model()          # no stamp -> fp32
    assert m.rc.amp_dtype is None
    assert _dtypes_seen(m, amp="bf16") == [torch.bfloat16]


def test_override_forces_fp32_on_a_bf16_checkpoint() -> None:
    m = _model(BF16_STAMP)
    assert m.rc.amp_dtype is torch.bfloat16
    assert _dtypes_seen(m, amp="32") == [torch.float32]
