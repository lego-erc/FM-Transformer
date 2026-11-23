"""What training reads from a whole dataset has the same values on a packed file as on a padded one, and is read
without materialising the padded tensors (``PackedData.data`` warns when it does)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import pytest
import torch
from torch.utils.data import random_split

from test_helpers import PACKED_COND, packed_prep, write_two_dirs
from legofmt.data.load.packed import load_dataset
from legofmt.data.load.loader import event_lengths as _event_lengths
from legofmt.data.load.loader import split_val
from legofmt.data.load.packed import interacting_only as _interacting_only
from legofmt.multiplicity.model import MultLoader


@contextmanager
def _no_materialising():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


def test_event_lengths_and_interacting_match(tmp_path) -> None:
    pad, pk, _ = write_two_dirs(tmp_path)
    a, b = (load_dataset(str(d), prep=packed_prep()) for d in (pad, pk))
    split = lambda ds: split_val(ds, {"val_frac": 0.34})[0]
    with _no_materialising():
        lb, ib = _event_lengths(split(b)), _interacting_only(b)
        lib = _event_lengths(ib)
    assert torch.equal(_event_lengths(split(a)), lb)
    ia = _interacting_only(a)
    assert torch.equal(ia.idx, ib.idx)
    assert torch.equal(_event_lengths(ia), lib)


def _mult_cfg(d, train_inverse: bool) -> dict:
    return {"mm_conf": {"max_out_particles": 8, "ptypes": torch.tensor([22.0, 2212.0]),
                        "ptypes_in": torch.tensor([2212.0]), "max_energy": 300.0, "cond_scalars": list(PACKED_COND),
                        "passthrough_head": True, "train_inverse": train_inverse},
            "dl_conf": {"lds_args": {"data": str(d), "cutoff_mev": 10.0}}}


@pytest.mark.parametrize("train_inverse", [False, True])
def test_multloader_matches(tmp_path, train_inverse) -> None:
    pad, pk, _ = write_two_dirs(tmp_path)
    a = MultLoader(_mult_cfg(pad, train_inverse))
    if train_inverse:
        b = MultLoader(_mult_cfg(pk, train_inverse))
    else:
        with _no_materialising():
            b = MultLoader(_mult_cfg(pk, train_inverse))
    names = ["in_tok", "counts", "pdgid_in_idx", "pt_target"]
    if train_inverse:
        names += ["out_tok", "out_pid_idx", "out_mask", "edep"]
    for n in names:
        assert torch.equal(getattr(a, n), getattr(b, n)), n
    assert b.counts.dtype == torch.int16 and b[[0, 1]][1].dtype == torch.int64   # stored small, batched as targets


def test_split_val_holds_index_tensors_of_random_splits_events() -> None:
    """``split_val`` holds the events ``random_split`` picks, as index tensors: ``random_split`` keeps a Python list,
    16M ints for the Ar cube, which every loader worker ends up copying."""
    ds = torch.arange(1000) * 3
    train, val = split_val(ds, {"val_frac": 0.1, "seed": 3})
    ref = random_split(ds, [900, 100], generator=torch.Generator().manual_seed(3))
    assert isinstance(train.idx, torch.Tensor) and isinstance(val.idx, torch.Tensor)
    assert train.idx.tolist() == ref[0].indices and val.idx.tolist() == ref[1].indices
    assert len(train) == 900 and torch.equal(train[[0, 5]], ds[train.idx[[0, 5]]]) and train[7] == ds[train.idx[7]]


def test_fresh_count_model_reads_the_dataset_once(tmp_path, monkeypatch) -> None:
    """Resolving a fresh config reads the dataset for its largest count, which sizes the count heads; ``setup`` must
    reuse that dataset, not read the file again."""
    from legofmt.multiplicity import model as mm
    _, pk, _ = write_two_dirs(tmp_path)
    built, init = [], mm.MultLoader.__init__
    monkeypatch.setattr(mm.MultLoader, "__init__", lambda self, c: (built.append(1), init(self, c))[1])
    model = mm.MultModel({"mm_conf": {"h_dim": 16, "n_layers": 1, "n_heads": 2},
                          "dl_conf": {"lds_args": {"data": str(pk)}, "bs": 2, "num_workers": 0}})
    model.setup("fit")
    assert len(built) == 1 and len(model._train_ds) + len(model._val_ds) == 6
    assert model.rc.max_particles == int(model._train_ds.ds.counts.max()) + 1


def test_training_step_identical(tmp_path) -> None:
    from test_helpers import _tiny_config
    from legofmt.main.modules import LEGOLtng
    pad, pk, _ = write_two_dirs(tmp_path)
    losses = []
    for d in (pad, pk):
        cfg = _tiny_config()
        cfg["config"]["dl_conf"]["lds_args"]["data"] = str(d)
        cfg["config"]["model_conf"]["model_args"]["max_seq_l"] = 10   # prepped_padded's L with 4 conditioning scalars
        torch.manual_seed(0)
        model = LEGOLtng(cfg)
        with _no_materialising():
            model.setup()
            batch = model.on_before_batch_transfer(next(iter(model.train_dataloader())), 0)
        model.on_fit_start()
        model.train()
        torch.manual_seed(1)
        losses.append(model._step(batch, 0))
    assert torch.isfinite(losses[0]) and torch.equal(losses[0], losses[1])
