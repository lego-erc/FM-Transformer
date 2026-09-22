"""model_conf.exclude_passthrough removes non-interacting events from the flow."""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import DataStruct, _F, set_layout
from legofmt.main.modules import _interacting_only


class _FakeDS:
    def __init__(self, data: DataStruct) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


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


def _edeps(view) -> list[float]:
    return [float(_F(view[i].f.full).edep.reshape(-1)) for i in range(len(view))]


def test_drops_only_zero_deposit_single_outgoing_events() -> None:
    view = _interacting_only(_dataset([0.0, 0.5, 0.0, 0.2], [1, 1, 2, 3]))
    assert len(view) == 3
    assert _edeps(view) == [0.5, 0.0, pytest.approx(0.2)]


def test_a_zero_deposit_event_with_secondaries_is_kept() -> None:
    """Zero deposit alone is not pass-through; the guard is n_out == 1 too."""
    assert len(_interacting_only(_dataset([0.0], [2]))) == 1


def test_the_surviving_edep_row_is_strictly_positive() -> None:
    view = _interacting_only(_dataset([0.0, 0.3, 0.0, 0.0, 1.0], [1, 1, 1, 1, 2]))
    assert all(e > 0 for e in _edeps(view))


def test_nothing_is_dropped_when_every_event_interacts() -> None:
    assert len(_interacting_only(_dataset([0.1, 0.2, 0.3], [1, 2, 1]))) == 3


def test_the_source_dataset_is_not_copied() -> None:
    """The view must alias the original storage, not duplicate 23 GB of it."""
    ds = _dataset([0.0, 0.5, 0.7], [1, 1, 1])
    view = _interacting_only(ds)
    assert view.ds is ds
    assert view[0].f.full.data_ptr() == ds.data[1].f.full.data_ptr()


def test_a_batch_of_indices_is_gathered_in_one_go() -> None:
    view = _interacting_only(_dataset([0.0, 0.5, 0.0, 0.2, 0.9], [1, 1, 1, 1, 1]))
    batch = view[torch.tensor([0, 2])]
    assert batch.f.full.shape[0] == 2
    assert _F(batch.f.full).edep.reshape(-1).tolist() == pytest.approx([0.5, 0.9])

