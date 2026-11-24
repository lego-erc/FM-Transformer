"""Exact PDG ids: a nucleus' 10-digit code is not exact in float32 (deuteron and triton both round to 1000010048), so a
packed file keeps every token's id as int32 beside its float32 tokens, and ``LEGOLtng.setup`` turns them into
vocabulary indices once (``rc.pdgid_is_idx``)."""

from __future__ import annotations

from dataclasses import replace

import torch

from test_helpers import _fake_batch, _tiny_config, prepped_padded
from legofmt.data.load.packed import PackedData
from legofmt.data.struct import DataStruct
from legofmt.data.write.pack import concat, pack
from legofmt.main.modules import LEGOLtng

D, T, ALPHA = 1000010020, 1000010030, 1000020040   # deuteron, triton, alpha
P = 2                                               # prefix tokens of ("Density",)


def _with_nuclei() -> tuple:
    """``prepped_padded`` in float64, with the first outgoing particle of every event that has one a nucleus."""
    f, m, am = prepped_padded()
    f = f.double()
    has = am[:, P + 1]
    f[has, P + 1, 7] = torch.tensor([D, T, ALPHA, D, T, ALPHA], dtype=torch.float64)[: int(has.sum())]
    return f, m, am


def test_float64_tokens_keep_exact_pdgids() -> None:
    f, m, am = _with_nuclei()
    pk = pack(f, m, am, n_prefix=P)
    assert pk.tokens.dtype == torch.float32 and pk.pdgid.dtype == torch.int32
    assert {D, T, ALPHA} <= set(pk.pdgid.tolist())
    assert torch.tensor(D, dtype=torch.float32) == torch.tensor(T, dtype=torch.float32)   # what float32 tokens did
    ref = pack(f.float(), m, am, n_prefix=P)   # everything else as the float32 packing of the same events
    assert ref.pdgid is None and torch.equal(pk.tokens.nan_to_num(-7), ref.tokens.nan_to_num(-7))


def test_pdgid_survives_dict_select_and_concat() -> None:
    f, m, am = _with_nuclei()
    pk = pack(f, m, am, n_prefix=P)
    assert torch.equal(PackedData.from_dict(pk.to_dict()).pdgid, pk.pdgid)
    idx = torch.tensor([4, 0, 1])
    assert torch.equal(pk.select(idx).pdgid, pack(f[idx], m[idx], am[idx], n_prefix=P).pdgid)
    parts = [pack(f[s], m[s], am[s], n_prefix=P) for s in (slice(0, 3), slice(3, 6))]
    assert torch.equal(concat(parts).pdgid, pk.pdgid)


def test_species_counts_tell_nuclei_apart() -> None:
    counts = pack(*_with_nuclei(), n_prefix=P).species_counts(torch.tensor([D, T]))
    assert counts[:, 0].sum() >= 1 and counts[:, 1].sum() >= 1   # in float32 both ids were one


def test_convert_pdgids_keeps_nuclei_of_the_vocabulary() -> None:
    """What setup does with a file's exact ids: a nucleus the vocabulary holds gets its own index, not 0."""
    model = LEGOLtng(_tiny_config(pdgids=torch.tensor([22, 211, 2212, D, T])))
    assert model.convert_pdgids(torch.tensor([22, D, T, ALPHA, 0])).tolist() == [1, 4, 5, 0, 0]


def test_index_mode_step_matches_raw_ids() -> None:
    """The same batch as raw ids and as vocabulary indices gives the same training loss."""
    model = LEGOLtng(_tiny_config())
    raw = _fake_batch()
    idx = DataStruct(raw.f.full.clone(), raw.m.full, raw.am.full)
    idx.f.full[..., 7] = model.convert_pdgids(raw.f.full[..., 7]).float()
    torch.manual_seed(0)
    loss_raw = model._step(raw, 0)
    model.rc = replace(model.rc, pdgid_is_idx=True)
    torch.manual_seed(0)
    loss_idx = model._step(idx, 0)
    assert torch.equal(loss_raw, loss_idx)
