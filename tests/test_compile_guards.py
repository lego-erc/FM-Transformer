"""The compiled ``ProjectModel.forward`` must not be reached under both
``d=None`` and ``d: Tensor``.

``torch.compile`` guards ``d is None`` separately from ``d: Tensor``, and that
axis multiplies with the ``grad_mode`` GLOBAL_STATE guard -- the CFM step and
the flow-map student run under grad, the flow-map teacher (``distill.py``) and
validation under ``no_grad``. Four graphs for one Python function, before the
train / last-partial / validation batch shapes are counted, which reaches the
default ``recompile_limit`` of 8. Exceeding it does not raise: dynamo drops the
frame to eager for the rest of the run, silently losing ``compile: model``
(observed 2026-09-04 on ``flow/kin_1step_040926``, rank 2, at 09:59).

So ``_step`` passes an explicit zero ``d`` whenever the field is
step-conditioned. ``ProjectModel.forward`` substitutes ``torch.zeros_like(t)``
for ``None``, so this is bitwise identical -- asserted below, because the whole
justification for the change is that it cannot alter the loss.
"""

from __future__ import annotations

import torch

from legofmt.main.modules import LEGOLtng
from test_modules_direct import _fake_batch, _tiny_config


def _model(step_cond: bool) -> LEGOLtng:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["model_args"]["step_cond"] = step_cond
    cfg["config"]["model_conf"]["base_pretrain_batches"] = 0
    return LEGOLtng(cfg)


def _args(m: LEGOLtng, B: int = 2, L: int = 5):
    torch.manual_seed(0)
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, 3:] = 1
    return (
        torch.randn(B, L, 7),
        torch.rand(B, 1),
        mask,
        torch.ones(B, L, dtype=torch.bool),
        torch.zeros(B, L, dtype=torch.long),
        torch.zeros(B, L, dtype=torch.long),
    )


def test_zero_d_is_bitwise_identical_to_none() -> None:
    m = _model(step_cond=True)
    m.model.eval()
    a = _args(m)
    with torch.no_grad():
        v_none = m.model(*a, d=None)
        v_zero = m.model(*a, d=torch.zeros_like(a[1]))
    assert torch.equal(v_none, v_zero), (v_none - v_zero).abs().max()


def test_step_passes_a_tensor_d_when_step_conditioned() -> None:
    """The guard axis is closed only if the CFM step passes a Tensor, not None."""
    m = _model(step_cond=True)
    seen: list[object] = []
    m.model.register_forward_pre_hook(
        lambda _mod, _a, kw: seen.append(kw.get("d", None)), with_kwargs=True,
    )
    m.train()
    m._step(_fake_batch(), 0)
    assert seen, "_step never called the model"
    assert isinstance(seen[0], torch.Tensor), f"CFM step passed d={seen[0]!r}"
    assert float(seen[0].abs().max()) == 0.0, "the CFM step's d must be zero"


def test_no_d_kwarg_when_not_step_conditioned() -> None:
    """Models without step_cond must keep passing None (vf never sees a d)."""
    m = _model(step_cond=False)
    assert getattr(m.model.vf, "step_cond", False) is False
