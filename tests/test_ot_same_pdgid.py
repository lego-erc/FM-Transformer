"""OT coupling is restricted to within one particle species, always.

The OT cost is species-blind, so the training assignment can hand a slot's
base draw to a *different* species -- systematically, since the leading
particle always claims the most compatible draw. The (species, draw) pairs
that never occur in training then occur constantly at sampling, where there is
no assignment, and the field acts ~identity there (per-species energy spectra
collapse toward the species-blind base; verified against abl_1gev_no_ot).
Cross-species pairs are blocked in the cost the same way
``inf_cond`` already blocks valid<->pad pairs.
"""

from __future__ import annotations

import itertools

import torch

from legofmt.base_dist import base_nn
from legofmt.data.struct import DataStruct, _F
from legofmt.main.modules import LEGOLtng
from test_modules_direct import _tiny_config


def _config(**model_conf) -> dict:
    cfg = _tiny_config()
    cfg["config"]["model_conf"].update(
        {"ot_coupling": True, "model_args": {**cfg["config"]["model_conf"]["model_args"], "max_seq_l": 7}},
        **model_conf,
    )
    return cfg


def _batch(B: int = 3, L: int = 7) -> DataStruct:
    """2 conditioning slots, 1 incoming, 4 outgoing: pdgids [22, 22, 211, 211]."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 0] = 0.1
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 2, 0] = 0.8
    f[:, 2, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 2, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f[:, 2, 7] = 22.0
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 0] = torch.rand(B, L - 3, generator=g)
    f[:, 3:, 1:4] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 4:7] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 7] = torch.tensor([22.0, 22.0, 211.0, 211.0])
    m = torch.zeros(B, L, dtype=torch.long)
    m[:, 1] = 1
    m[:, 3:] = 1
    am = torch.ones(B, L, dtype=torch.bool)
    return DataStruct(f, m, am)


class _BruteLap:
    """Exact CPU stand-in for torch_lap_cuda's solve_lap; records the cost."""

    def __init__(self) -> None:
        self.cost = None

    def __call__(self, cost: torch.Tensor, device) -> torch.Tensor:
        self.cost = cost.detach().clone()
        B, n, _ = cost.shape
        out = torch.empty(B, n, dtype=torch.long)
        for b in range(B):
            perms = list(itertools.permutations(range(n)))
            totals = [sum(cost[b, i, p[i]].item() for i in range(n)) for p in perms]
            out[b] = torch.tensor(perms[int(torch.tensor(totals).argmin())])
        return out


def _run(model_conf: dict) -> tuple[_BruteLap, torch.Tensor]:
    lap = _BruteLap()
    orig = base_nn.slap
    base_nn.slap = lap
    try:
        model = LEGOLtng(_config(**model_conf))
        model.on_fit_start()
        model.train()
        torch.manual_seed(0)
        model.gen_base_wrapper(_batch())
    finally:
        base_nn.slap = orig
    pid = _batch().f.out_p[..., -1]
    return lap, pid


def test_cost_blocks_cross_species_pairs() -> None:
    lap, pid = _run({})
    cross = pid.unsqueeze(-1) != pid.unsqueeze(-2)
    assert lap.cost is not None, "OT branch never ran"
    assert (lap.cost[cross] >= 1e5).all(), "cross-species pair not blocked"
    assert (lap.cost[~cross] < 1e5).all(), "within-species pair wrongly blocked"


def test_assignment_stays_within_species() -> None:
    lap, pid = _run({})
    B, n, _ = lap.cost.shape
    assign = _BruteLap.__call__(lap, lap.cost, "cpu")
    picked = torch.take_along_dim(pid, assign, dim=1)
    assert torch.equal(picked, pid), f"assignment crossed species:\n{pid}\n{picked}"

