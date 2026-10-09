"""Inference runs in the precision the checkpoint was trained with.

``run.precision`` is a Lightning *training* autocast: weights are saved in fp32
and, before this, inference ran fp32 no matter how the model was trained. That
left a measured 1.6x on the table for bf16-trained checkpoints -- the per-token
conditional-linear GEMM is 38% of inference CUDA time and runs on FP32 units
(profiled at 2-step midpoint, N=16383: 0.1426 -> 0.0898 ms/event under bf16
autocast, MMD vs Geant4 0.0321 -> 0.0286, i.e. no accuracy cost).

``scripts/train.py`` already stamps ``additional.precision``; ``amp_dtype``
derives the autocast dtype from it. Checkpoints with no stamp (every pre-2026-09
one) resolve to ``None`` and keep running fp32, so nothing changes for them.

The dangerous case is handled elsewhere and deliberately not re-gated here:
``compat.fp32_attention`` already forces attention back to fp32 whenever
``attn_qk_norm_scale > 100``, i.e. exactly the pre-2.25.5 checkpoints whose
attention logits are ``1000*cos`` and where bf16 rounding is O(1).
"""

from __future__ import annotations

import torch

from legofmt.mod_comps.config import _amp_dtype


def test_bf16_variants_map_to_bfloat16() -> None:
    for s in ("bf16-mixed", "bf16", "bf16-true", "bf16-mixed, medium"):
        assert _amp_dtype(s) is torch.bfloat16, s


def test_fp16_variants_map_to_float16() -> None:
    for s in ("16-mixed", "16", "16-true", "16-mixed, high"):
        assert _amp_dtype(s) is torch.float16, s


def test_fp32_and_unknown_and_missing_disable_autocast() -> None:
    for s in ("32", "32, medium", "64", "transformer-engine", "", None, 32):
        assert _amp_dtype(s) is None, repr(s)


def test_resolved_config_carries_amp_dtype() -> None:
    from test_modules_direct import _tiny_config
    from legofmt.mod_comps.config import resolve_legoltng_config

    cfg = _tiny_config()
    assert resolve_legoltng_config(cfg).amp_dtype is None, "no stamp => fp32"

    cfg = _tiny_config()
    cfg["config"]["additional"] = {"precision": "bf16-mixed, medium"}
    assert resolve_legoltng_config(cfg).amp_dtype is torch.bfloat16
