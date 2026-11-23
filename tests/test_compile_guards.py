"""``_step`` hands a field with ``step_cond`` a zero tensor as ``d``, never ``None``.

``torch.compile`` guards ``d is None`` apart from a tensor ``d``, and that guard multiplies with the grad-mode guard
(the CFM step and the flow-map student run under grad, the teacher and validation under ``no_grad``): four graphs of
one function before the batch shapes are counted. That reaches dynamo's default ``recompile_limit`` of 8, and past it
dynamo silently runs the frame eager for the rest of the run (seen in ``flow/kin_1step_040926``).
``ProjectModel`` replaces a ``None`` by ``zeros_like(t)``, so the zero tensor gives bitwise the same velocity: the
first test asserts it.
"""

from __future__ import annotations

import torch

from legofmt.main.modules import LEGOLtng
from test_helpers import _fake_batch, _tiny_config


def _model(step_cond: bool) -> LEGOLtng:
    return LEGOLtng(_tiny_config({"step_cond": step_cond}))


def _args(B: int = 2, L: int = 5):
    torch.manual_seed(0)
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, 3:] = 1
    return (torch.randn(B, L, 7), torch.rand(B, 1), mask, torch.ones(B, L, dtype=torch.bool),
            torch.zeros(B, L, dtype=torch.long), torch.zeros(B, L, dtype=torch.long))


def test_zero_d_is_bitwise_identical_to_none() -> None:
    m = _model(step_cond=True)
    m.model.eval()
    a = _args()
    with torch.no_grad():
        v_none = m.model(*a, d=None)
        v_zero = m.model(*a, d=torch.zeros_like(a[1]))
    assert torch.equal(v_none, v_zero), (v_none - v_zero).abs().max()


def test_step_passes_a_tensor_d_when_step_conditioned() -> None:
    """The CFM step passes ``d`` as a zero tensor; a ``None`` would add the ``d is None`` guard and its graphs."""
    m = _model(step_cond=True)
    seen: list[object] = []
    m.model.register_forward_pre_hook(lambda _mod, _a, kw: seen.append(kw.get("d", None)), with_kwargs=True)
    m.train()
    m._step(_fake_batch(), 0)
    assert seen, "_step never called the model"
    assert isinstance(seen[0], torch.Tensor), f"CFM step passed d={seen[0]!r}"
    assert float(seen[0].abs().max()) == 0.0, "the CFM step's d must be zero"


def test_no_d_kwarg_when_not_step_conditioned() -> None:
    """A model without ``step_cond`` builds a field without it: that field never reads a ``d``."""
    m = _model(step_cond=False)
    assert getattr(m.model.vf, "step_cond", False) is False
