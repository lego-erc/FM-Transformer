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
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng
from test_helpers import _fake_batch, _tiny_config


def _batch() -> DataStruct:
    """``_fake_batch`` with 4 outgoing slots, pdgids [22, 22, 211, 211], and the E_dep row generated."""
    ds = _fake_batch(B=3, L=7, edep_generated=True)
    ds.f.full[:, 3:, 7] = torch.tensor([22.0, 22.0, 211.0, 211.0])
    return ds


class _BruteLap:
    """Exact CPU stand-in for torch_linear_assignment's batch_linear_assignment; records the cost."""

    def __init__(self) -> None:
        self.cost = None

    def __call__(self, cost: torch.Tensor) -> torch.Tensor:
        self.cost = cost.detach().clone()
        B, n, _ = cost.shape
        out = torch.empty(B, n, dtype=torch.long)
        for b in range(B):
            perms = list(itertools.permutations(range(n)))
            totals = [sum(cost[b, i, p[i]].item() for i in range(n)) for p in perms]
            out[b] = torch.tensor(perms[int(torch.tensor(totals).argmin())])
        return out


def _run(monkeypatch) -> tuple[_BruteLap, torch.Tensor]:
    lap = _BruteLap()
    monkeypatch.setattr(base_nn, "batch_linear_assignment", lap)
    model = LEGOLtng(_tiny_config({"max_seq_l": 7}, ot_coupling=True))
    model.on_fit_start()
    model.train()
    torch.manual_seed(0)
    model.gen_base_wrapper(_batch())
    return lap, _batch().f.out_p[..., -1]


def test_cost_blocks_cross_species_pairs(monkeypatch) -> None:
    lap, pid = _run(monkeypatch)
    cross = pid.unsqueeze(-1) != pid.unsqueeze(-2)
    assert lap.cost is not None, "OT branch never ran"
    assert (lap.cost[cross] >= 1e5).all(), "cross-species pair not blocked"
    assert (lap.cost[~cross] < 1e5).all(), "within-species pair wrongly blocked"


def test_assignment_stays_within_species(monkeypatch) -> None:
    lap, pid = _run(monkeypatch)
    picked = torch.take_along_dim(pid, lap(lap.cost), dim=1)
    assert torch.equal(picked, pid), f"assignment crossed species:\n{pid}\n{picked}"
