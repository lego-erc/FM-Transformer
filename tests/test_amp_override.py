"""``odeint_conf["amp"]`` overrides the checkpoint's precision stamp.

``amp_dtype`` is derived from ``additional.precision``, which makes the solve's
precision a property of how the checkpoint was *trained*. It is not: a
2x2 of checkpoint against inference dtype (80 NFE, rho=3, N=16383) put the
fp32-trained checkpoint at MMD 0.0100 in fp32 and 0.0097 under a bf16
autocast, and the bf16-trained one at 0.0108 / 0.0111 -- i.e. the ~1.5x is
available to either, and belongs to the solver settings next to
``fwd_compile``, not to the stamp.

So ``forward()`` prefers ``odeint_conf["amp"]`` when the key is present. The
distinction between *absent* and *explicitly None* matters: absent means "use
the stamp", ``None`` means "force fp32 on a bf16-trained checkpoint".
"""

from __future__ import annotations

import torch

from legofmt.mod_comps.config import _amp_dtype


def test_dtype_passes_through() -> None:
    assert _amp_dtype(torch.bfloat16) is torch.bfloat16
    assert _amp_dtype(torch.float16) is torch.float16
    assert _amp_dtype(torch.float32) is None


def test_strings_still_resolve() -> None:
    assert _amp_dtype("bf16") is torch.bfloat16
    assert _amp_dtype("bf16-mixed, medium") is torch.bfloat16
    assert _amp_dtype("16-mixed") is torch.float16
    for s in ("32", "32, medium", None, ""):
        assert _amp_dtype(s) is None, repr(s)


def _dtypes_seen(model, cfg_amp: object, sentinel: object) -> list[torch.dtype]:
    """Run one solve and report the autocast dtype the vector field ran under."""
    from legofmt.data.struct import DataStruct
    from test_modules_direct import _fake_batch

    if sentinel is not None:            # `None` here means "leave the key out"
        model.rc.odeint_conf["amp"] = cfg_amp
    model.rc.odeint_conf.update({"method": "euler", "step_size": 1.0})

    seen: list[torch.dtype] = []
    hook = model.model.register_forward_pre_hook(
        lambda *_a, **_k: seen.append(
            torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu")
            else torch.float32),
    )
    try:
        model.eval()
        model(_fake_batch())
    finally:
        hook.remove()
    assert seen, "the vector field was never called"
    return seen


def test_absent_key_uses_the_stamp() -> None:
    from legofmt.main.modules import LEGOLtng
    from test_modules_direct import _tiny_config

    cfg = _tiny_config()
    cfg["config"]["additional"] = {"precision": "bf16-mixed, medium"}
    m = LEGOLtng(cfg)
    assert m.rc.amp_dtype is torch.bfloat16
    assert "amp" not in m.rc.odeint_conf, "fixture must not preset the key"


def test_override_forces_bf16_on_an_fp32_checkpoint() -> None:
    from legofmt.main.modules import LEGOLtng
    from test_modules_direct import _tiny_config

    m = LEGOLtng(_tiny_config())          # no stamp -> fp32
    assert m.rc.amp_dtype is None
    assert _dtypes_seen(m, "bf16", sentinel=True) == [torch.bfloat16]


def test_override_forces_fp32_on_a_bf16_checkpoint() -> None:
    from legofmt.main.modules import LEGOLtng
    from test_modules_direct import _tiny_config

    cfg = _tiny_config()
    cfg["config"]["additional"] = {"precision": "bf16-mixed, medium"}
    m = LEGOLtng(cfg)
    assert m.rc.amp_dtype is torch.bfloat16
    assert _dtypes_seen(m, "32", sentinel=True) == [torch.float32]
