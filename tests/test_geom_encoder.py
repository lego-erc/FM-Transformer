"""model_args.geom_encoder: K learned tokens in the flow's h_dim that summarise a layered stack.

A small bidirectional transformer over one token per layer ([thickness mm, density g/cm3, Z, A], beam side first,
plus radiation / interaction lengths and Fourier features of the cumulative depth) prepends K summary tokens to the
flow's sequence. They are conditioning tokens (t = 1, d = 0), never transported and dropped before the output
projection. With ``hide_scalars`` the composite Density / Z / A / Size rows stay in the data (the base head and
the count model read them) but are masked out of the flow's attention, so the stack reaches the flow only through
the tokens.
"""

from __future__ import annotations

import copy

import pytest
import torch

from legofmt.cfm.geom_encoder import GeomEncoder
from legofmt.main.modules import LEGOLtng
from legofmt.mod_comps.config import resolve_legoltng_config

from test_helpers import GEOM, _fake_batch, _tiny_config, _vf_kwargs

W, SI = GEOM["layers"]


def _config(geom=GEOM) -> dict:
    return _tiny_config(None if geom is None else {"geom_encoder": copy.deepcopy(geom)})


def _model(geom=GEOM, seed=0) -> LEGOLtng:
    torch.manual_seed(seed)
    m = LEGOLtng(_config(geom))
    m.on_fit_start()
    m.eval()
    return m


def _wake(m: LEGOLtng) -> None:
    """The encoder's output layer is zero-init (the tokens start as plain conditioning tokens); give it weights so
    a test can see what the tokens carry."""
    torch.manual_seed(1)
    torch.nn.init.normal_(m.model.vf.geom.out.weight, std=0.5)


def _velocity(m: LEGOLtng, ds, t=0.3):
    return m.model(ds.f.model_in, torch.tensor(t), **_vf_kwargs(m, ds))


def test_config_expands_the_repeated_period_and_records_the_prefix() -> None:
    rc = resolve_legoltng_config(_config())
    g = rc.model_args["geom_encoder"]
    assert torch.tensor(g["layers"]).shape == (6, 4) and "repeat" not in g
    assert g["n_prefix"] == rc.n_prefix and g["hide_scalars"] is True


def test_bad_layers_raise() -> None:
    with pytest.raises(ValueError, match="geom_encoder"):
        resolve_legoltng_config(_config({**GEOM, "layers": [[1.4, 19.3, 74.0]]}))


def test_tokens_have_the_flow_width_and_see_the_layer_order() -> None:
    enc = GeomEncoder([W, SI] * 3, h_dim=16, tokens=4, dim=16, depth=1, heads=2)
    assert enc().shape == (1, 4, 16)
    assert enc().abs().max() == 0                       # zero-init read-out: tokens start empty
    torch.nn.init.normal_(enc.out.weight)
    rev = GeomEncoder([SI, W] * 3, h_dim=16, tokens=4, dim=16, depth=1, heads=2)
    rev.load_state_dict(enc.state_dict())
    assert not torch.allclose(enc(), rev())


def test_output_shape_and_conditioning_rows_unchanged() -> None:
    m = _model()
    _wake(m)
    ds = _fake_batch()
    v = _velocity(m, ds)
    assert v.shape == ds.f.full.shape[:2] + (7,)
    cond_rows = ds.m.full == 0
    assert v[cond_rows].abs().max() == 0


@pytest.mark.parametrize("hide", [True, False])
def test_hidden_scalar_rows_do_not_reach_the_generated_rows(hide: bool) -> None:
    m = _model({**GEOM, "hide_scalars": hide})
    _wake(m)
    ds = _fake_batch()
    f2 = ds.f.full.clone()
    f2[:, 0, 0] = 7.0                                  # the Density row
    ds2 = type(ds)(f2, ds.m.full, ds.am.full)
    gen = ds.m.full == 1
    same = torch.allclose(_velocity(m, ds)[gen], _velocity(m, ds2)[gen], atol=1e-6)
    assert same is hide


def test_tokens_change_the_field() -> None:
    m = _model()
    ds = _fake_batch()
    v0 = _velocity(m, ds)                               # zero tokens
    _wake(m)
    assert not torch.allclose(v0, _velocity(m, ds))


def test_packed_and_per_token_conditions_agree() -> None:
    """Per-event t uses the packed (B, 3, D) adaptive-norm condition; its generated-token selector must skip the K
    prefix tokens. A per-token t equal to it (conditioning tokens at t = 1) takes the unpacked path."""
    m = _model()
    _wake(m)
    ds = _fake_batch()
    vf, ex = m.model.vf, _vf_kwargs(m, ds)
    x = ds.f.model_in
    t_ev = torch.full((2, 1), 0.3)
    t_tok = torch.where(ds.m.full == 1, 0.3, 1.0)
    torch.testing.assert_close(vf(x, t=t_ev, **ex), vf(x, t=t_tok, **ex), rtol=1e-5, atol=1e-6)


def test_training_step_reaches_the_encoder() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_config())
    m.on_fit_start()
    m.train()
    loss = m._step(_fake_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.model.vf.geom.out.weight.grad.abs().sum() > 0


def test_checkpoint_round_trips(tmp_path) -> None:
    m = _model()
    _wake(m)
    path = tmp_path / "flow.pt"
    torch.save({"state_dict": m.model.vf.state_dict(), "config": m.rc.config}, path)
    m2 = LEGOLtng(torch.load(path, weights_only=False))
    m2.on_fit_start()
    m2.eval()
    ds = _fake_batch()
    torch.testing.assert_close(_velocity(m, ds), _velocity(m2, ds))


def test_without_the_encoder_nothing_changes() -> None:
    m = _model(geom=None)
    assert getattr(m.model.vf, "geom", None) is None
    assert not any("geom" in k for k in m.model.vf.state_dict())
