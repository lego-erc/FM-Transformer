"""``model_args.cond_pool``: the cond scalars reach the field as one token instead of one each.

A set encoder over (which scalar, standardised value) pairs: ``rho(sum_k phi([z_k, onehot_k]))``. The scalars' own
tokens leave the sequence the encoder runs, 3 of an Ar cube event's 18 real tokens on average. In the A/B of
20.10.2026 this form matched the per-scalar tokens in loss and in interpolation between trained densities; Fourier
features of the value did not.
"""

from __future__ import annotations

import pytest
import torch

from test_helpers import _head_batch, _head_config
from legofmt.config import resolve_legoltng_config
from legofmt.main.modules import LEGOLtng

N_PREFIX = 5        # Density, E_dep, Z, A, Size


def _model(**model_args) -> LEGOLtng:
    torch.manual_seed(0)
    model = LEGOLtng(_head_config(True, {"cond_pool": True, "dropout": 0.0, "ntypes": N_PREFIX + 2, **model_args}))
    with torch.no_grad():   # the read-out is not zero-initialised, but a scale the tests can see helps
        model.model.vf.cond_pool.mu.copy_(torch.tensor([3.0, 18.0, 40.0, 100.0]))
    return model.eval()


def _velocity(model: LEGOLtng, ds, L: int | None = None) -> torch.Tensor:
    L = L or ds.f.full.shape[1]
    t = torch.full((len(ds), 1), 0.3)
    return model.model(ds.f.model_in[:, :L], t, ds.m.full[:, :L], ds.am.full[:, :L], model.types_embd[:, :L],
                       model.convert_pdgids(ds.f.pdgids)[:, :L])


def _batch():
    return _head_batch(torch.tensor([0.5, 0.2, 0.4, 0.1]), torch.tensor([1, 2, 2, 1]))


def test_config_hands_the_field_the_prefix_length() -> None:
    rc = resolve_legoltng_config(_head_config(True, {"cond_pool": True}))
    assert rc.model_args["cond_pool"] == N_PREFIX
    assert "cond_pool" not in resolve_legoltng_config(_head_config(True)).model_args


def test_the_encoder_runs_one_token_for_the_scalars() -> None:
    model, ds, seen = _model(), _batch(), []
    model.model.vf.vf.attn_layers.register_forward_pre_hook(lambda mod, args: seen.append(args[0].shape[1]))
    v = _velocity(model, ds)
    assert seen == [ds.f.full.shape[1] - (N_PREFIX - 1) + 1]       # the 4 scalar tokens out, their pooled token in
    assert v.shape == ds.f.model_in.shape and torch.isfinite(v).all()
    assert (v[ds.m.full != 1] == 0).all() and v[ds.m.full == 1].abs().sum() > 0


def test_the_pooled_token_carries_the_scalars_values() -> None:
    model, ds = _model(), _batch()
    v = _velocity(model, ds)
    denser = ds.f.full.clone()
    denser[:, 0, 0] = 7.0                                           # Density
    other = type(ds)(denser, ds.m.full, ds.am.full)
    assert (_velocity(model, other) - v).abs().max() > 1e-4
    filler = ds.f.full.clone()
    filler[:, 2, 1:7] = 0.3                                         # the Z token's other channels are not read
    assert torch.equal(_velocity(model, type(ds)(filler, ds.m.full, ds.am.full)), v)


def test_velocity_is_unchanged_by_trimming_the_padding() -> None:
    model = _model()
    ds = _head_batch(torch.tensor([0.5, 0.2]), torch.tensor([1, 1]))     # 7 real tokens of 8
    full, trimmed = _velocity(model, ds), _velocity(model, ds, L=7)
    torch.testing.assert_close(trimmed, full[:, :7], rtol=1e-5, atol=1e-6)


def test_a_step_trains_the_pool_and_generation_runs() -> None:
    model = _model().train()
    model.on_fit_start()
    model._step(_batch(), 0).backward()
    pool = model.model.vf.cond_pool
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in pool.parameters())
    out = model.eval()(_batch())
    assert out[0].shape[:2] == (4, 8)


def test_the_scale_is_the_datas_and_is_saved_with_the_field() -> None:
    pool = _model().model.vf.cond_pool
    values = torch.tensor([[1.0, 18.0, 40.0, 100.0], [3.0, 29.0, 63.5, 100.0], [5.0, 18.0, 40.0, 100.0]])
    pool.standardise_by(values)
    assert torch.allclose(pool.mu, values.mean(0)) and torch.allclose(pool.sd[:3], values.std(0)[:3])
    assert pool.sd[3] == 1.0   # a scalar the dataset holds fixed (one cube size): its own unit, not a division by zero
    assert {"cond_pool.mu", "cond_pool.sd"} <= set(_model().model.vf.state_dict())
    assert not any(k.startswith("cond_pool") for k in LEGOLtng(_head_config(True)).model.vf.state_dict())


@pytest.mark.parametrize("clash", [{"geom_encoder": {"layers": [[1.0, 2.0, 3.0, 4.0]]}}, {"time_cond": False}],
                         ids=["geom_encoder", "time_cond_off"])
def test_cond_pool_refuses_what_it_does_not_cover(clash) -> None:
    with pytest.raises(ValueError, match="cond_pool"):
        resolve_legoltng_config(_head_config(True, {"cond_pool": True, **clash}))
