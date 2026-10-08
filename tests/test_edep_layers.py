"""Optional per-layer E_dep: ``per_event["E_dep_layers"]`` of shape (B, n_layers).

Prep carries it as a 4th tensor next to (f, m, am), in MeV like the E_dep row;
``norm_e`` puts it on the E_dep row's model scale at load time. The token layout
and the models are untouched, and 3-tuple files load as before.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.prep import DataPrep
from legofmt.data.struct import DataStruct

CUTOFF = 10.0
MANIFOLD = [{"name": "euclidean", "dim": 1},
            {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}]


def _cfg(**kw):
    return {"cutoff_mev": CUTOFF, "manifold": MANIFOLD, "proj_ray": False,
            "cond_scalars": ["Density"], "energy_kin": True, **kw}


def _raw(n=4, k=3):
    """n events, event 1 entirely below the cutoff (dropped by prep)."""
    def row(e, d, p, pid):
        return [e, *[e * c for c in d], *p, pid]
    inc = torch.tensor([[row(200.0, (1, 0, 0), (-50, 0, 0), 2212.0)]]).repeat(n, 1, 1)
    out = torch.tensor([[row(50.0, (0, 1, 0), (0, 50, 0), 22.0)]]).repeat(n, 2, 1)
    inc[1, :, 0] = out[1, :, 0] = 1.0
    layers = torch.arange(n * k, dtype=torch.float32).view(n, k) + 1.0
    return {"per_particle": {"Incoming": inc, "Outgoing": out},
            "per_event": {"E_dep": layers.sum(-1), "Density": torch.full((n,), 3.0),
                          "E_dep_layers": layers}}


def test_prep_carries_edep_layers_of_the_kept_events() -> None:
    raw = _raw()
    d = LEGODataset(data=raw, prep=DataPrep(_cfg()), cutoff_mev=CUTOFF).data
    keep = torch.tensor([0, 2, 3])
    assert len(d) == 3
    torch.testing.assert_close(d.edep_layers, raw["per_event"]["E_dep_layers"][keep])
    torch.testing.assert_close(d.f.edep, raw["per_event"]["E_dep"][keep])   # E_dep row unchanged


@pytest.mark.parametrize("cfg", [_cfg(max_energy=300.0),
                                 _cfg(max_energy=300.0, edep_log_min=1e-3)])
def test_load_puts_edep_layers_on_the_edep_scale(tmp_path, cfg) -> None:
    raw = _raw()
    d = LEGODataset(data=raw, prep=DataPrep(cfg), cutoff_mev=CUTOFF).data
    torch.save((d.f.full, d.m.full, d.am.full, d.edep_layers), tmp_path / "data_prepped.pt")
    prep = DataPrep(cfg)
    loaded = LEGODataset(data=str(tmp_path / "data_prepped.pt"), prep=prep).data
    torch.testing.assert_close(loaded.edep_layers, prep.norm_edep(d.edep_layers))
    torch.testing.assert_close(loaded.f.edep, prep.norm_edep(d.f.edep))


def test_datastruct_keeps_edep_layers_aligned() -> None:
    d = LEGODataset(data=_raw(), prep=DataPrep(_cfg()), cutoff_mev=CUTOFF).data
    idx = torch.tensor([2, 0])
    torch.testing.assert_close(d[idx].edep_layers, d.edep_layers[idx])
    torch.testing.assert_close(d.to("cpu").edep_layers, d.edep_layers)
    batch = default_collate([d[0], d[2]])
    torch.testing.assert_close(batch.edep_layers, d.edep_layers[[0, 2]])


def test_without_edep_layers_nothing_changes(tmp_path) -> None:
    raw = _raw()
    del raw["per_event"]["E_dep_layers"]
    d = LEGODataset(data=raw, prep=DataPrep(_cfg()), cutoff_mev=CUTOFF).data
    assert d.edep_layers is None
    torch.save((d.f.full, d.m.full, d.am.full), tmp_path / "data_prepped.pt")
    loaded = LEGODataset(data=str(tmp_path / "data_prepped.pt"), prep=DataPrep(_cfg(max_energy=300.0))).data
    assert loaded.edep_layers is None
    assert default_collate([loaded[0], loaded[1]]).edep_layers is None
    assert DataStruct(d.f.full, d.m.full, d.am.full).edep_layers is None
