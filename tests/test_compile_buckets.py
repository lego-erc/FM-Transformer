"""odeint_conf.compile_buckets: pad every solve chunk to one of a few batch
sizes so a compiled velocity model only ever sees that many shapes.

The composition chain hands the flow a different particle count every round;
without this, inductor recompiles per round until the dynamo cache limit drops
it to eager (~135 s per leg for ~5 s of generation).
"""

from __future__ import annotations

import pytest
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


@pytest.mark.parametrize("n, split_size, buckets, shapes", [
    (5, None, None, [5]),          # no buckets: the old behaviour
    (5, None, [8, 16], [8]),       # a partial chunk is padded to the smallest fitting bucket
    (8, None, [8, 16], [8]),       # an exact bucket size is not padded
    (20, 16, [8, 16], [16, 8]),    # split, then bucket: exactly the bucket shapes
    (20, 16, [8], [16, 8]),        # a chunk larger than every bucket is left alone
], ids=["no_buckets", "partial_padded", "exact_not_padded", "split_then_bucket", "larger_than_buckets"])
def test_chunks_run_at_their_bucket_and_come_back_unpadded(n, split_size, buckets, shapes) -> None:
    x, m = torch.randn(n, 3, 2), torch.ones(n, 3)
    fn, seen = _recording(lambda x, m: x * 2)
    out = _solvers().chunked(fn, x, m, split_size=split_size, buckets=buckets)
    assert seen == [(k, k) for k in shapes]
    assert torch.equal(out, x * 2)


def test_padding_tiles_the_chunks_own_rows() -> None:
    """Padded rows must be valid events, not zeros or garbage."""
    s = _solvers()
    x = torch.arange(3.0).view(3, 1, 1)
    captured = []
    s.chunked(lambda x: captured.append(x) or x, x, buckets=[8])
    assert captured[0].squeeze().tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]


def test_stacked_intermediates_are_sliced_on_the_batch_axis() -> None:
    """cat_dim=-3 is the batch axis of a (T, B, L, C) intermediates stack."""
    s = _solvers()
    x = torch.randn(5, 3, 2)
    out = s.chunked(lambda x: torch.stack([x, x * 2]), x, cat_dim=-3, buckets=[8])
    assert out.shape == (2, 5, 3, 2)
    assert torch.equal(out[1], x * 2)


def test_forward_reads_compile_buckets_from_odeint_conf() -> None:
    import inspect

    src = inspect.getsource(Solvers._forward)   # forward is the no_grad wrapper around it
    assert 'cfg.get("compile_buckets")' in src
