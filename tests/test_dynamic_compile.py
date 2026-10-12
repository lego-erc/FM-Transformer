"""``odeint_conf.fwd_compile`` compiles the field as ONE symbolic-shape graph.

``torch.compile(dynamic=True)`` alone is not enough: x-transformers masks the
attention output with ``einx.where('b n, b n d, -> b n d', mask, out, 0.)`` and
einx traces per concrete shape, which specialises batch and length, so every
bucket shape of the count-mode solve got its own graph (~9 s each, on every
call whose bucket counts differ). ``compat.dynamic_shape_attention`` swaps that
one pattern for the equivalent ``torch.where`` (measured 2026-10-12: 15-22 s
compile once, then any shape at no cost, 2.5x over eager at 16k events).
"""

from __future__ import annotations

import inspect

import torch
import x_transformers.x_transformers as xt

from legofmt.cfm.solvers import Solvers
from legofmt.compat import dynamic_shape_attention


def test_patched_where_matches_einx_and_delegates_other_patterns() -> None:
    import einx

    dynamic_shape_attention()
    dynamic_shape_attention()                       # idempotent
    mask, out = torch.rand(2, 5) > 0.5, torch.randn(2, 5, 3)
    assert torch.equal(xt.einx.where("b n, b n d, -> b n d", mask, out, 0.0),
                       einx.where("b n, b n d, -> b n d", mask, out, 0.0))
    a, b = torch.randn(4), torch.randn(3)
    assert torch.equal(xt.einx.subtract("j, i -> i j", a, b), einx.subtract("j, i -> i j", a, b))
    assert xt.einx.rearrange is einx.rearrange   # everything else is einx's own


def test_forward_compiles_one_dynamic_graph() -> None:
    src = inspect.getsource(Solvers.forward)
    assert "dynamic_shape_attention()" in src
    assert "dynamic=True" in src and "dynamic=False" not in src
