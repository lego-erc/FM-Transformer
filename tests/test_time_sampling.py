"""Tests for the training time ``t`` (``main/train_step.py``): SD3 mode sampling and the power shift ``t_dist_shift``."""

from unittest.mock import patch

import pytest
import torch

from test_helpers import _fake_batch, _tiny_config
from legofmt.main.modules import LEGOLtng


def _times(**model_conf) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the times a tiny flow samples for 4096 events with seed 0, and the uniform draws ``u`` behind them."""
    model, ds = LEGOLtng(_tiny_config(**model_conf)), _fake_batch(B=4096)
    torch.manual_seed(0)
    t = model._sample_t(ds)
    torch.manual_seed(0)
    return t, torch.rand_like(ds.f.d)


@pytest.mark.parametrize("k", [1.0, 0.7, 1.3])
def test_shift_is_a_power_of_the_sd3_time(k) -> None:
    """``t_dist_shift`` k maps the SD3 time to ``t^(1/k)``; with scale 0 the SD3 time is ``1 - u``. k = 1 changes
    nothing, k < 1 moves the mass toward t = 0."""
    t, u = _times(t_dist="sd3", t_dist_scale=0.0, t_dist_shift=k)
    assert torch.equal(t, (1 - u) ** (1 / k) if k != 1.0 else 1 - u)


def test_shift_runs_in_the_recipe_path() -> None:
    """The shifted SD3 time is the recipe's own path, not the legacy sampler's."""
    with patch.object(LEGOLtng, "_sample_t_options", side_effect=AssertionError("legacy path")):
        _times(t_dist="sd3", t_dist_scale=0.0, t_dist_shift=0.7)


def test_shift_applies_to_the_legacy_sm_norm_time() -> None:
    """The logit-normal time (legacy ``sm_norm``) takes the same shift."""
    model, ds = LEGOLtng(_tiny_config(t_dist="sm_norm", t_dist_scale=1.0, t_dist_shift=0.7)), _fake_batch(B=4096)
    torch.manual_seed(0)
    t = model._sample_t(ds)
    torch.manual_seed(0)
    assert torch.equal(t, torch.sigmoid(torch.randn_like(ds.f.d)) ** (1 / 0.7))


@pytest.mark.parametrize("k", [0.0, -0.5])
def test_shift_must_be_positive(k) -> None:
    """A shift k <= 0 sends the time outside [0, 1] (``t^(1/k) >= 1``)."""
    with pytest.raises(ValueError, match="t_dist_shift"):
        LEGOLtng(_tiny_config(t_dist_shift=k))
