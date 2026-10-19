"""``odeint_conf.compile_dynamic`` compiles the field as ONE symbolic-shape graph.

``torch.compile(dynamic=True)`` alone is not enough: x-transformers masks the
attention output with ``einx.where('b n, b n d, -> b n d', mask, out, 0.)`` and
einx traces per concrete shape, which specialises batch and length, so every
(length bucket, row bucket) shape still got its own graph. ``compat.dynamic_shape_attention``
swaps that one pattern for the equivalent ``torch.where`` (feat/unify-models, measured
2026-10-12: 15-22 s compile once, then any shape at no cost). The static grid compiled
28 graphs in ~5 min for the 59x W/Si slab chain (2026-10-16).
"""

from __future__ import annotations

import pytest
import torch
import x_transformers.x_transformers as xt

from legofmt.compat import dynamic_shape_attention
from test_len_buckets import _config, _model, _ragged_batch


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


@pytest.mark.parametrize("dynamic, buckets, mode", [(False, [2, 4, 8], "reduce-overhead"), (False, None, "reduce-overhead"),
                                                   (True, [2, 4, 8], "reduce-overhead"), (True, None, None)])
def test_forward_compiles_static_or_dynamic(monkeypatch, dynamic: bool, buckets, mode) -> None:
    """CUDA graphs only where the shapes are bounded: always for the static grid, for a dynamic graph only with
    compile_buckets (every new row count would record and keep another graph)."""
    seen = {}
    monkeypatch.setattr(torch, "compile", lambda m, **k: seen.update(k) or m)
    model = _model(_config(odeint={"method": "euler", "step_size": 0.25, "fwd_compile": True,
                                   "compile_dynamic": dynamic, "compile_buckets": buckets}))
    model(_ragged_batch())
    assert seen["dynamic"] is dynamic and seen["mode"] == mode
    assert getattr(xt.einx, "_legofmt_where", False) or not dynamic


def test_dynamic_graph_sees_no_shape_it_would_specialise_on(monkeypatch) -> None:
    """Dynamo still guards a symbolic graph on a 1-row batch (0/1 specialisation) and on the strides / storage
    offset of its inputs; length buckets and split chunks are views. Either recompiles the whole flow (16 graphs and
    ~4 min of compile for the W/Si slab chain, 2026-10-17), so the field must see >= 2 contiguous rows."""
    monkeypatch.setattr(torch, "compile", lambda m, **k: m)
    model = _model(_config(odeint={"method": "euler", "step_size": 0.25, "fwd_compile": True, "compile_dynamic": True,
                                   "len_buckets": [4, 6], "split_size": 2}))
    seen = []
    model.model.register_forward_pre_hook(lambda mod, args, kw: seen.append(
        (args[0].shape[0], all(a.is_contiguous() for a in (*args, *kw.values()) if torch.is_tensor(a)))),
        with_kwargs=True)
    model(_ragged_batch(n_out=(1, 2, 5, 3, 1)))            # lengths 4,5,8,6,4 -> one event alone in bucket 8
    assert seen and all(n >= 2 for n, _ in seen), seen
    assert all(c for _, c in seen), seen
