"""odeint_conf.compile_buckets: pad every solve chunk to one of a few batch
sizes so a compiled velocity model only ever sees that many shapes.

The composition chain hands the flow a different particle count every round;
without this, inductor recompiles per round until the dynamo cache limit drops
it to eager (~135 s per leg for ~5 s of generation).
"""

from __future__ import annotations

import torch

from legofmt.cfm.solvers import Solvers


def _solvers() -> Solvers:
    return Solvers.__new__(Solvers)


def _recording(fn):
    seen = []

    def wrapped(*tensors):
        seen.append(tuple(t.shape[0] for t in tensors))
        return fn(*tensors)

    return wrapped, seen


def test_no_buckets_is_the_old_behaviour() -> None:
    s = _solvers()
    x, m = torch.randn(5, 3, 2), torch.ones(5, 3)
    fn, seen = _recording(lambda x, m: x * 2)
    out = s.chunked(fn, x, m, split_size=None, buckets=None)
    assert seen == [(5, 5)]
    assert torch.equal(out, x * 2)


def test_partial_chunk_is_padded_to_the_smallest_fitting_bucket() -> None:
    s = _solvers()
    x, m = torch.randn(5, 3, 2), torch.ones(5, 3)
    fn, seen = _recording(lambda x, m: x * 2)
    out = s.chunked(fn, x, m, buckets=[8, 16])
    assert seen == [(8, 8)]
    assert out.shape[0] == 5
    assert torch.equal(out, x * 2)


def test_exact_bucket_size_is_not_padded() -> None:
    s = _solvers()
    x, m = torch.randn(8, 3, 2), torch.ones(8, 3)
    fn, seen = _recording(lambda x, m: x)
    s.chunked(fn, x, m, buckets=[8, 16])
    assert seen == [(8, 8)]


def test_padding_tiles_the_chunks_own_rows() -> None:
    """Padded rows must be valid events, not zeros or garbage."""
    s = _solvers()
    x = torch.arange(3.0).view(3, 1, 1)
    captured = []
    s.chunked(lambda x: captured.append(x) or x, x, buckets=[8])
    assert captured[0].squeeze().tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]


def test_split_then_bucket_gives_exactly_the_bucket_shapes() -> None:
    s = _solvers()
    x, m = torch.randn(20, 3, 2), torch.ones(20, 3)
    fn, seen = _recording(lambda x, m: x * 3)
    out = s.chunked(fn, x, m, split_size=16, buckets=[8, 16])
    assert seen == [(16, 16), (8, 8)]
    assert out.shape[0] == 20
    assert torch.equal(out, x * 3)


def test_chunk_larger_than_every_bucket_is_left_alone() -> None:
    s = _solvers()
    x = torch.randn(20, 3, 2)
    fn, seen = _recording(lambda x: x)
    s.chunked(fn, x, split_size=16, buckets=[8])
    assert seen == [(16,), (8,)]


def test_stacked_intermediates_are_sliced_on_the_batch_axis() -> None:
    """cat_dim=-3 is the batch axis of a (T, B, L, C) intermediates stack."""
    s = _solvers()
    x = torch.randn(5, 3, 2)
    out = s.chunked(lambda x: torch.stack([x, x * 2]), x, cat_dim=-3, buckets=[8])
    assert out.shape == (2, 5, 3, 2)
    assert torch.equal(out[1], x * 2)


def test_forward_reads_compile_buckets_from_odeint_conf() -> None:
    import inspect

    src = inspect.getsource(Solvers.forward)
    assert 'cfg.get("compile_buckets")' in src
