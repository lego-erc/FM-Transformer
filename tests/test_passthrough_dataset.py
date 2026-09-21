"""model_conf.exclude_passthrough removes non-interacting events from the flow."""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import DataStruct, _F, set_layout
from legofmt.main.modules import _drop_passthrough


class _FakeDS:
    def __init__(self, data: DataStruct) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)


def _dataset(edeps: list[float], n_outs: list[int]) -> _FakeDS:
    set_layout(("Density",))
    n_prefix, n_out_slots = 2, 3
    L = n_prefix + 1 + n_out_slots
    f = torch.zeros(len(edeps), L, 8)
    am = torch.zeros(len(edeps), L)
    am[:, : n_prefix + 1] = 1
    for i, (e, k) in enumerate(zip(edeps, n_outs)):
        f[i, 1, 0] = e
        am[i, n_prefix + 1 : n_prefix + 1 + k] = 1
    return _FakeDS(DataStruct(f, torch.ones_like(am).long(), am))


def test_drops_only_zero_deposit_single_outgoing_events() -> None:
    ds = _dataset([0.0, 0.5, 0.0, 0.2], [1, 1, 2, 3])
    _drop_passthrough(ds)
    assert len(ds) == 3
    assert _F(ds.data.f.full).edep.reshape(-1).tolist() == pytest.approx([0.5, 0.0, 0.2])


def test_a_zero_deposit_event_with_secondaries_is_kept() -> None:
    """Zero deposit alone is not pass-through; the guard is n_out == 1 too."""
    ds = _dataset([0.0], [2])
    _drop_passthrough(ds)
    assert len(ds) == 1


def test_the_surviving_edep_row_is_strictly_positive() -> None:
    ds = _dataset([0.0, 0.3, 0.0, 0.0, 1.0], [1, 1, 1, 1, 2])
    _drop_passthrough(ds)
    assert (_F(ds.data.f.full).edep > 0).all()


def test_nothing_is_dropped_when_every_event_interacts() -> None:
    ds = _dataset([0.1, 0.2, 0.3], [1, 2, 1])
    _drop_passthrough(ds)
    assert len(ds) == 3
