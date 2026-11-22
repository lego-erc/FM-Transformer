"""``load_dataset`` on a packed file gives exactly the padded file's batches.

Same events, two files: everything a caller can index or ask for must be bit-identical.
"""

from __future__ import annotations

import json
import warnings

import pytest
import torch
from legofmt.data.load.loader import split_val

from test_helpers import packed_prep, write_two_dirs
from legofmt.data.load.packed import load_dataset
from legofmt.data.load.packed import PackedData


@pytest.fixture
def two_dirs(tmp_path):
    return write_two_dirs(tmp_path)


def _same(x, y) -> None:
    for u, v in ((x.f.full, y.f.full), (x.m.full, y.m.full), (x.am.full, y.am.full)):
        assert torch.equal(u, v)
    assert (x.edep_layers is None) == (y.edep_layers is None)
    if x.edep_layers is not None:
        assert torch.equal(x.edep_layers, y.edep_layers)


def test_dispatch(two_dirs) -> None:
    pad, pk, both = two_dirs
    assert not isinstance(load_dataset(str(pad)), PackedData)
    assert isinstance(load_dataset(str(pk)), PackedData)
    assert isinstance(load_dataset(str(both)), PackedData)
    assert not isinstance(load_dataset(str(both / "data_prepped.pt")), PackedData)
    assert isinstance(load_dataset(str(both / "data_packed.pt")), PackedData)


@pytest.mark.parametrize("prep_kw", [None, {}, {"edep_log_min": 1e-3}])
@pytest.mark.parametrize("frac", [None, 0.5])
def test_batches_equal_padded(two_dirs, prep_kw, frac) -> None:
    pad, pk, _ = two_dirs
    prep = None if prep_kw is None else packed_prep(**prep_kw)
    a, b = (load_dataset(str(d), prep=prep, frac=frac) for d in (pad, pk))
    assert len(a) == len(b)
    for idx in (torch.tensor([0, 2]), [1, 1, 2], torch.arange(len(a)), torch.tensor([], dtype=torch.long), 1,
                torch.tensor(1), [], -1, torch.tensor([-1, 0]), [-2, 1], slice(1, 3),
                torch.arange(len(a)) % 2 == 0):
        _same(a[idx], b[idx])


def test_bad_indices_raise_like_padded(two_dirs) -> None:
    pad, pk, _ = two_dirs
    a, b = load_dataset(str(pad)), load_dataset(str(pk))
    for idx in (torch.tensor([]), torch.tensor([1.0, 2.0]), len(a), -len(a) - 1, torch.tensor([0, len(a)])):
        for ds in (a, b):
            with pytest.raises(IndexError):
                ds[idx]


def test_stale_packed_file_is_refused(two_dirs) -> None:
    _, _, both = two_dirs
    meta = json.loads((both / "meta.json").read_text())
    (both / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "format"}))
    with pytest.raises(ValueError, match="stale"):          # a padded writer rewrote meta.json after packing
        load_dataset(str(both))
    assert not isinstance(load_dataset(str(both / "data_prepped.pt")), PackedData)
    assert isinstance(load_dataset(str(both / "data_packed.pt")), PackedData)


def test_subset_chain(two_dirs) -> None:
    pad, pk, _ = two_dirs
    subs = [split_val(load_dataset(str(d), prep=packed_prep()), {"val_frac": 0.34}) for d in (pad, pk)]
    for (ta, va), (tb, vb) in [subs]:
        _same(ta[[0, 3, 1]], tb[[0, 3, 1]])
        _same(va[[1, 0]], vb[[1, 0]])


def test_accessors_match_padded(two_dirs) -> None:
    pad, pk, _ = two_dirs
    a, b = (load_dataset(str(d), prep=packed_prep()) for d in (pad, pk))
    ptypes = torch.tensor([22.0, 2212.0])
    for get in (lambda d: d.lengths, lambda d: d.n_out, lambda d: d.edep, lambda d: d.cond("Density"),
                lambda d: d.cond("Size"), lambda d: d.incoming, lambda d: d.species_counts(ptypes),
                lambda d: d.edep_layers):
        assert torch.equal(get(a), get(b))


def test_data_property_materialises_once(two_dirs) -> None:
    pad, pk, _ = two_dirs
    b = load_dataset(str(pk), prep=packed_prep())
    with pytest.warns(UserWarning, match="GB"):
        d = b.data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert b.data is d
    _same(d, load_dataset(str(pad), prep=packed_prep()).data)


def test_packed_needs_a_file_and_a_norm_e_prep(two_dirs) -> None:
    _, pk, _ = two_dirs
    with pytest.raises(TypeError):
        load_dataset(torch.load(pk / "data_packed.pt", weights_only=False))
    with pytest.raises(TypeError):
        load_dataset(str(pk), prep=lambda batch: batch)
