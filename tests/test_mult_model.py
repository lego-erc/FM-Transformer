"""MultModel fixes from the 2026-10-17 review."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from legofmt.multiplicity.model import MultModel
from test_helpers import _mult_config


def _in(B: int = 4):
    in_tok = torch.zeros(B, 8)
    in_tok[:, 1] = 0.5
    in_tok[:, 4] = 1.0
    in_tok[:, 7] = -1.0
    return in_tok, None, torch.zeros(B, dtype=torch.long)


def test_schedule_free_is_back_in_train_mode_after_validation() -> None:
    """forward() switches the optimizer to eval; without the validation hooks the
    next optimizer step raised 'Optimizer was not in train mode' at step 0."""
    cfg = _mult_config()  # opt: schedulefree
    m = MultModel({"state_dict": {}, "config": cfg})
    m._trainer = SimpleNamespace(model=m)
    m.on_fit_start()
    m.on_validation_model_eval()
    m(_in())
    m.on_validation_model_train()
    assert all(g["train_mode"] for g in m.opt.param_groups)


def test_missing_checkpoint_keys_warn() -> None:
    sd = MultModel({"state_dict": {}, "config": _mult_config()}).state_dict()
    sd.pop("proj_out_b")
    with pytest.warns(UserWarning, match="proj_out_b"):
        MultModel({"state_dict": sd, "config": _mult_config()})
