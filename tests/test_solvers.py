"""The hand-unrolled solvers transport only ``mask == 1`` slots and stay on the manifold.

Conditioning and pad rows must come back bit-identical to ``x_init`` (a solver once
moved them), and the sphere factors must stay unit: ``rk4`` used to run through
flow_matching's Euclidean ``ODESolver``, which let them drift off the sphere.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.main.modules import LEGOLtng
from test_helpers import _fake_batch, _tiny_config


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
def test_solver_keeps_conditioning_rows_and_the_manifold(method: str) -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_tiny_config())
    with torch.no_grad():  # a fast field, so drift off the sphere would be visible
        m.model.vf.cond_w_pdgids[:, 1].mul_(50.0)
    ds = _fake_batch(B=8)
    ds.am.full[1, -1] = False; ds.m.full[1, -1] = 0  # one pad slot, masked like real padding
    x0 = m.gen_base_wrapper(ds)
    out = m.solve(ds, x_init=x0, method=method, step_size=0.25)

    gen = (ds.m.full == 1) & ds.am.full
    assert torch.equal(out[~gen], x0.where(ds.am.full.unsqueeze(-1), x0[:, 2:3])[~gen])
    assert not torch.equal(out[gen], x0[gen]), "nothing was transported"
    for sl in (slice(1, 4), slice(4, 7)):
        norm = out[..., sl].norm(dim=-1)[gen]
        assert (norm - 1).abs().max() < 1e-5, (method, (norm - 1).abs().max())


def test_rk4_intermediates_follow_the_time_grid() -> None:
    m = LEGOLtng(_tiny_config())
    ds = _fake_batch()
    xs = m.solve(ds, method="rk4", step_size=0.25, return_intermediates=True)
    assert xs.shape[0] == 5  # x_0 and one state per step, as for euler / midpoint
