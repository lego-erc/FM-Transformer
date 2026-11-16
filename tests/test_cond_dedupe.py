"""Per-event (packed) conditioning reproduces the per-token computation.

``to_gamma`` is zero-initialised, so at init every condition gives the same gamma
and any packing bug is invisible; the weights are randomised here first.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x


def _vf(step_cond: bool) -> CFMTrafo_x:
    torch.manual_seed(0)
    vf = CFMTrafo_x(h_dim=16, nhead=2, max_seq_l=6, ntypes=4, in_dim=7, nlayers=2, npdgids=3, dropout=0.0,
                    step_cond=step_cond, use_adaptive_rmsnorm=True, use_adaptive_layerscale=True,
                    ff_glu=True, ff_swish=True).double()
    with torch.no_grad():
        for p in vf.parameters():
            p.add_(0.3 * torch.randn_like(p))
    assert vf._packed_cond
    return vf.eval()


@pytest.mark.parametrize("step_cond", [False, True])
def test_per_event_condition_matches_per_token(step_cond: bool) -> None:
    B, L = 5, 6
    vf = _vf(step_cond)
    x = torch.randn(B, L, 7, dtype=torch.float64)
    mask = torch.randint(0, 2, (B, L))
    mask[0] = 0  # an event with nothing generated
    am = torch.ones(B, L, dtype=torch.bool)
    am[1, -1] = False
    types = torch.arange(L).clamp_max(3).view(1, -1)
    pdg = torch.randint(0, 3, (B, L, 1))
    t, d = torch.rand(B, 1, dtype=torch.float64), torch.rand(B, 1, dtype=torch.float64)
    gen = mask == 1
    per_event = vf(x, mask, am, types, pdg, t=t, d=d if step_cond else None)
    per_token = vf(x, mask, am, types, pdg, t=torch.where(gen, t, 1.0),
                   d=torch.where(gen, d, 0.0) if step_cond else None)
    torch.testing.assert_close(per_event, per_token, rtol=0, atol=1e-12)
    scalar = vf(x, mask, am, types, pdg, t=t[:1, :1], d=d[:1, :1] if step_cond else None)  # solver form
    torch.testing.assert_close(scalar, vf(x, mask, am, types, pdg, t=torch.where(gen, t[0, 0], 1.0),
                                          d=torch.where(gen, d[0, 0], 0.0) if step_cond else None),
                               rtol=0, atol=1e-12)
