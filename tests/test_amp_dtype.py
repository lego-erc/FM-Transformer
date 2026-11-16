"""Inference autocast dtype: the checkpoint's precision stamp, overridable per solve.

``run.precision`` is a Lightning *training* autocast; ``scripts/train.py`` stamps it as ``additional.precision`` and
``amp_dtype`` derives the inference autocast from it. Before, inference ran fp32 however the model was trained, which
left 1.6x on the table for bf16-trained checkpoints: the per-token conditional-linear GEMM is 38% of inference CUDA
time (2-step midpoint, N=16383: 0.1426 -> 0.0898 ms/event under bf16, MMD vs Geant4 0.0321 -> 0.0286). Unstamped
(pre-2026-09) checkpoints resolve to ``None`` and keep running fp32.

The speed-up is not a property of how the checkpoint was trained (80 NFE, rho=3, N=16383: fp32-trained MMD 0.0100 fp32
/ 0.0097 bf16, bf16-trained 0.0108 / 0.0111), so ``odeint_conf["amp"]`` overrides the stamp when present: absent
means "use the stamp", "32" / ``None`` forces fp32 on a bf16-trained checkpoint. Pre-2.25.5 checkpoints, whose
attention logits are ``1000*cos``, are kept fp32 elsewhere (``compat.fp32_attention``) and not re-gated here.
"""

from __future__ import annotations

import torch

from legofmt.main.modules import LEGOLtng
from legofmt.mod_comps.config import _amp_dtype, resolve_legoltng_config
from test_helpers import _fake_batch, _tiny_config

BF16_STAMP = {"precision": "bf16-mixed, medium"}


def test_precision_specs_map_to_autocast_dtypes() -> None:
    for want, specs in ((torch.bfloat16, ("bf16-mixed", "bf16", "bf16-true", "bf16-mixed, medium", torch.bfloat16)),
                        (torch.float16, ("16-mixed", "16", "16-true", "16-mixed, high", torch.float16)),
                        (None, ("32", "32, medium", "64", "transformer-engine", "", None, 32, torch.float32))):
        for s in specs:
            assert _amp_dtype(s) is want, repr(s)


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
    """Run one solve (``amp`` merged into odeint_conf) and report the autocast dtype the vector field ran under."""
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
