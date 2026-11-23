"""The OT coupling pairs base and data slots within one species only.

The cost itself does not see the species. Unrestricted, the assignment hands a slot's base sample to a slot of
another species, and systematically so, since the leading particle claims the most compatible sample. The
(species, base sample) pairs that training then never shows are the rule at generation, where there is no
assignment, and the field is close to the identity on them: the per-species energy spectra collapse towards the
species-blind base. Cross-species pairs are therefore blocked in the cost, like the pairs of a particle with padding.
"""

from __future__ import annotations

import itertools

import torch

from legofmt.base import ot_coupling
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng
from test_helpers import _fake_batch, _tiny_config


def _batch() -> DataStruct:
    """Returns ``_fake_batch`` with 4 outgoing slots of pdgids [22, 22, 211, 211] and a generated E_dep token."""
    ds = _fake_batch(B=3, L=7, edep_generated=True)
    ds.f.full[:, 3:, 7] = torch.tensor([22.0, 22.0, 211.0, 211.0])
    return ds


class _BruteLap:
    """An exact CPU stand-in for torch_linear_assignment's ``batch_linear_assignment`` that records the cost."""

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
    monkeypatch.setattr(ot_coupling, "batch_linear_assignment", lap)
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
