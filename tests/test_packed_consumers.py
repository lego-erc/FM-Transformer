"""The training-side readers of a whole dataset give the same values on a packed file,
without materialising its padded tensors (``LEGODataset.data`` warns when it does)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import pytest
import torch
from torch.utils.data import random_split

from test_helpers import PACKED_COND, packed_prep, write_two_dirs
from legofmt.data.dataloaders import LEGODataset
from legofmt.main.modules import _event_lengths, _interacting_only
from legofmt.multiplicity.model import MultLoader


@contextmanager
def _no_materialising():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


def test_event_lengths_and_interacting_match(tmp_path) -> None:
    pad, pk, _ = write_two_dirs(tmp_path)
    a, b = (LEGODataset(str(d), prep=packed_prep()) for d in (pad, pk))
    split = lambda ds: random_split(ds, [4, 2], generator=torch.Generator().manual_seed(0))[0]
    with _no_materialising():
        lb, ib = _event_lengths(split(b)), _interacting_only(b)
        lib = _event_lengths(ib)
    assert torch.equal(_event_lengths(split(a)), lb)
    ia = _interacting_only(a)
    assert torch.equal(ia.keep_idx, ib.keep_idx)
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


def test_training_step_identical(tmp_path) -> None:
    from test_helpers import _tiny_config
    from legofmt.main.modules import LEGOLtng
    pad, pk, _ = write_two_dirs(tmp_path)
    losses = []
    for d in (pad, pk):
        cfg = _tiny_config()
        cfg["config"]["dl_conf"]["lds_args"]["data"] = str(d)
        cfg["config"]["model_conf"]["model_args"]["max_seq_l"] = 10   # prepped_padded's L with 4 cond scalars
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
