"""The packed dataset format: only the real rows of each event plus where it starts.

``PackedData`` must give back exactly the padded tensors it was packed from -- every
bit, padded rows included -- for any set of events.
"""

from __future__ import annotations

import pytest
import torch

from test_helpers import prepped_padded
from legofmt.data.packed import PACKED_FORMAT, PACKED_VERSION, PackedData, is_packed

CONDS = [("Density",), ("Density", "Z", "A", "Size")]


def _pack(t, cond, layers=False) -> PackedData:
    return PackedData.from_padded(*t[:3], n_prefix=len(cond) + 1, edep_layers=t[3] if layers else None)


def _equal(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("cond", CONDS)
@pytest.mark.parametrize("layers", [False, True])
def test_round_trip_is_bit_exact(cond, layers) -> None:
    t = prepped_padded(cond, layers)
    assert _equal(_pack(t, cond, layers).unpack(), t)


@pytest.mark.parametrize("cond", CONDS)
@pytest.mark.parametrize("idx", [torch.tensor([3, 0, 3]), torch.tensor([], dtype=torch.long)])
def test_unpack_index_equals_fancy_index(cond, idx) -> None:
    t = prepped_padded(cond, layers=True)
    assert _equal(_pack(t, cond, True).unpack(idx), tuple(x[idx] for x in t))


@pytest.mark.parametrize("cond", CONDS)
def test_block_without_padding_concats(cond) -> None:
    t, n_prefix = prepped_padded(cond), len(cond) + 1
    full = t[2].sum(1) == t[0].shape[1]
    a = PackedData.from_padded(*(x[full] for x in t), n_prefix=n_prefix)
    assert torch.isnan(a.pad_row).all()
    b = PackedData.from_padded(*(x[~full] for x in t), n_prefix=n_prefix)
    out = PackedData.concat([a, b]).unpack()
    assert _equal(out, tuple(torch.cat((x[full], x[~full])) for x in t))


@pytest.mark.parametrize("cond", CONDS)
def test_concat_equals_packing_whole(cond) -> None:
    t = prepped_padded(cond, layers=True)
    parts = [_pack(tuple(x[s] for x in t), cond, True) for s in (slice(0, 2), slice(2, 4), slice(4, 6))]
    whole, joined = _pack(t, cond, True).to_dict(), PackedData.concat(parts).to_dict()
    assert whole.keys() == joined.keys()
    for k, v in whole.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v.nan_to_num(-7.0), joined[k].nan_to_num(-7.0)), k
        else:
            assert v == joined[k], k


@pytest.mark.parametrize("cond", CONDS)
def test_select_equals_unpack_of_index(cond) -> None:
    p, idx = _pack(prepped_padded(cond, layers=True), cond, True), torch.tensor([5, 1, 2])
    assert _equal(p.select(idx).unpack(), p.unpack(idx))


@pytest.mark.parametrize("cond", CONDS)
def test_dict_round_trip(cond) -> None:
    t = prepped_padded(cond, layers=True)
    d = _pack(t, cond, True).to_dict()
    assert d["format"] == PACKED_FORMAT == "lego-packed" and d["version"] == PACKED_VERSION == 1
    assert is_packed(d) and not is_packed(t) and not is_packed({"per_event": {}})
    assert _equal(PackedData.from_dict(d).unpack(), t)
    with pytest.raises(ValueError):
        PackedData.from_dict({**d, "version": 2})


@pytest.mark.parametrize("cond", CONDS)
def test_accessors(cond) -> None:
    t, n_prefix = prepped_padded(cond), len(cond) + 1
    p = _pack(t, cond)
    assert len(p) == t[0].shape[0]
    assert torch.equal(p.lengths, t[2].sum(1))
    assert torch.equal(p.n_out, t[2][:, n_prefix + 1:].sum(1))
    assert torch.equal(p.incoming, t[0][:, n_prefix])
    pid = t[0][:, n_prefix + 1:, -1]
    ptypes = torch.tensor([22.0, 2212.0])
    assert torch.equal(p.species_counts(ptypes), torch.stack([(pid == q).sum(1) for q in ptypes], -1))
    with pytest.raises(ValueError):
        p.species_counts(torch.tensor([0.0, 22.0]))
    assert p.rows.shape[0] == int(t[2][:, n_prefix:].sum())


@pytest.mark.parametrize("cond", CONDS)
def test_from_padded_rejects_unrepresentable(cond) -> None:
    t, n_prefix = prepped_padded(cond), len(cond) + 1
    f, m, am = t
    bad_am = am.clone()
    bad_am[0, n_prefix + 1] = False                     # event 0 is full length: a gap before later rows
    bad_m = m.clone()
    bad_m[0, 0] = 1
    pad = (~am).nonzero()[0]
    bad_pad = f.clone()
    bad_pad[pad[0], pad[1], 3] += 1.0
    bad_prefix = f.clone()
    bad_prefix[1, 0, 2] += 1.0
    for args in ((f, m, bad_am), (f, bad_m, am), (bad_pad, m, am), (bad_prefix, m, am)):
        with pytest.raises(ValueError):
            PackedData.from_padded(*args, n_prefix=n_prefix)
    with pytest.raises(TypeError):
        PackedData.from_padded(f, m.bool(), am, n_prefix=n_prefix)
