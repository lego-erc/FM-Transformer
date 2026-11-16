"""model_conf.dir_zoom: outgoing momentum directions are learned with their polar angle about the incoming
direction remapped by theta' = pi * log1p(theta / theta_s) / log1p(pi / theta_s).

A chained thin plate needs survivors resolved to ~1 mrad (G4: a 1 GeV proton scatters 7 mrad in 1.4 mm W, 0.5 mrad
in 0.3 mm Si); the flow resolves ~10 mrad on the plain sphere and the error random-walks over the stack. The remap
is a bijection of the sphere that magnifies small deflections, applied to the targets in training and undone on the
generated sample.
"""

from __future__ import annotations

import math

import pytest
import torch

from legofmt.data.struct import DataStruct
from legofmt.geometry.geom_trafos import zoom_polar
from legofmt.main.modules import LEGOLtng
from legofmt.mod_comps.config import resolve_legoltng_config

from test_helpers import _fake_batch, _tiny_config

THETA_S = 2e-3


def _pairs(n: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=torch.float64), dim=-1)
    w = torch.nn.functional.normalize(torch.linalg.cross(u, torch.randn(n, 3, generator=g, dtype=torch.float64)),
                                      dim=-1)
    theta = torch.cat((torch.logspace(-6, 0, n // 2, dtype=torch.float64),
                       torch.rand(n - n // 2, generator=g, dtype=torch.float64) * math.pi))
    return u, torch.cos(theta)[:, None] * u + torch.sin(theta)[:, None] * w


def test_zoom_round_trips() -> None:
    u, v = _pairs()
    back = zoom_polar(zoom_polar(v, u, THETA_S), u, THETA_S, inverse=True)
    assert (back - v).abs().max() < 1e-9


def test_zoom_maps_the_polar_angle_and_keeps_the_azimuth() -> None:
    u, v = _pairs()
    z = zoom_polar(v, u, THETA_S)
    theta = torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), (u * v).sum(-1))
    theta_z = torch.atan2(torch.linalg.cross(u, z).norm(dim=-1), (u * z).sum(-1))
    expect = math.pi * torch.log1p(theta / THETA_S) / math.log1p(math.pi / THETA_S)
    assert torch.allclose(theta_z, expect, atol=1e-9)
    plane = torch.linalg.cross(u, v)
    assert (plane * z).sum(-1).abs().max() < 1e-9                                  # in the (u, v) plane
    assert ((torch.linalg.cross(u, z) * plane).sum(-1) >= -1e-12).all()            # on v's side of u
    assert torch.allclose(zoom_polar(u, u, THETA_S), u)                            # the incoming direction is fixed


def _model(**gate) -> LEGOLtng:
    torch.manual_seed(0)
    return LEGOLtng(_tiny_config(dir_zoom=THETA_S, **gate))


GATED = {"dir_zoom_nucleons_max": 1}


def _nucleon_batch(model: LEGOLtng) -> DataStruct:
    """Event 0 emits two protons (a nuclear interaction), event 1 a proton and a photon (a survivor)."""
    ds = _fake_batch()
    f, n = ds.f.full.clone(), model.rc.n_prefix
    f[0, n + 1:, 7] = 2212.0
    f[1, n + 1:, 7] = torch.tensor([2212.0, 22.0])
    return DataStruct(f, ds.m.full, ds.am.full)


def test_config_resolves_the_dir_zoom_keys() -> None:
    off = resolve_legoltng_config(_tiny_config())
    assert off.dir_zoom is None and off.dir_zoom_mix == 0.0 and off.dir_zoom_nucleons_max is None
    assert _model().rc.dir_zoom == THETA_S and _model(**GATED).rc.dir_zoom_nucleons_max == 1
    assert resolve_legoltng_config(_tiny_config(dir_zoom=THETA_S, dir_zoom_mix=0.5)).dir_zoom_mix == 0.5
    with pytest.raises(ValueError, match="dir_zoom_nucleons_max"):
        resolve_legoltng_config(_tiny_config(dir_zoom=THETA_S, dir_zoom_nucleons_max=-1))


def test_only_outgoing_rows_of_forward_events_are_zoomed() -> None:
    model = _model()
    ds = _fake_batch()
    x, am = ds.f.full, ds.am.full
    fwd = torch.ones(len(ds), dtype=torch.bool)
    fwd[0] = False
    z = model._zoom_dirs(x, am, fwd)
    n = model.rc.n_prefix
    out_rows = am.clone()
    out_rows[:, :n + 1] = False
    out_rows[0] = False
    assert torch.equal(z[~out_rows], x[~out_rows])
    assert torch.equal(z[out_rows][:, [0, 4, 5, 6, 7]], x[out_rows][:, [0, 4, 5, 6, 7]])
    assert not torch.allclose(z[out_rows][:, 1:4], x[out_rows][:, 1:4])


def test_events_with_more_nucleons_than_the_gate_are_not_zoomed() -> None:
    """Nuclear secondaries are spread over the sphere; the zoom is for near-identity survivors and squeezes them. In
    G4, >= 2 outgoing nucleons marks a nuclear interaction in 99.4-99.9% of them (1.4 mm W, 0.1-1 GeV n/p); survivors
    and EM events have <= 1 in 99.9%+."""
    model = _model(**GATED)
    ds = _nucleon_batch(model)
    x, n = ds.f.full, model.rc.n_prefix
    z = model._zoom_dirs(x, ds.am.full, torch.ones(len(ds), dtype=torch.bool), pdgids=ds.f.pdgids)
    assert torch.equal(z[0], x[0])
    assert not torch.allclose(z[1, n + 1:, 1:4], x[1, n + 1:, 1:4])


@pytest.mark.parametrize("gate", [{}, GATED], ids=["plain", "gated"])
def test_training_step_learns_the_zoomed_targets(gate) -> None:
    """The base, the OT coupling and the flow all see the zoomed outgoing directions."""
    model = _model(**gate)
    seen = []
    orig = model.gen_base_wrapper
    model.gen_base_wrapper = lambda ds_t, *a, **k: seen.append(ds_t.f.full.clone()) or orig(ds_t, *a, **k)
    ds = _nucleon_batch(model) if gate else _fake_batch()
    expect = model._zoom_dirs(ds.f.full, ds.am.full, ds.m.full[:, model.rc.n_prefix] == 0, pdgids=ds.f.pdgids)
    model._step(ds, 0)
    keep = ds.am.full
    assert torch.allclose(seen[0][..., 1:4][keep], expect[..., 1:4][keep])


@pytest.mark.parametrize("gate", [{}, GATED], ids=["plain", "gated"])
def test_forward_undoes_the_zoom_on_the_generated_sample(gate) -> None:
    """The solver works in zoomed coordinates; forward hands back physical directions."""
    model = _model(**gate).eval()
    ds = _nucleon_batch(model) if gate else _fake_batch()
    target = ds.f.model_in.clone()
    fwd = ds.m.full[:, model.rc.n_prefix] == 0
    zoomed = model._zoom_dirs(target, ds.am.full, fwd, pdgids=ds.f.pdgids)
    model._solve_by_length = lambda *a, **k: zoomed.clone()
    sols, _, _ = model(ds)
    keep = ds.am.full
    assert torch.allclose(sols[..., 1:4][keep], target[..., 1:4][keep], atol=1e-5)


@pytest.mark.parametrize("mix", [0.25, 0.5])
def test_blended_zoom_round_trips_and_keeps_the_backward_hemisphere_wide(mix) -> None:
    """dir_zoom_mix w blends the linear map in: theta' = w theta + (1 - w) pi log1p(theta/theta_s)/log1p(pi/theta_s).
    The pure log puts the whole backward hemisphere into the last 9% of theta' (theta_s 2e-3), where the flow's error
    is magnified ~4x in physical angle; nuclear secondaries going backwards came out 2-4x too often."""
    u, v = _pairs()
    z = zoom_polar(v, u, THETA_S, mix=mix)
    theta = torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), (u * v).sum(-1))
    theta_z = torch.atan2(torch.linalg.cross(u, z).norm(dim=-1), (u * z).sum(-1))
    expect = mix * theta + (1 - mix) * math.pi * torch.log1p(theta / THETA_S) / math.log1p(math.pi / THETA_S)
    assert torch.allclose(theta_z, expect, atol=1e-9)
    assert (zoom_polar(z, u, THETA_S, inverse=True, mix=mix) - v).abs().max() < 1e-9
    back = mix * math.pi / 2 + (1 - mix) * math.pi * math.log1p(math.pi / 2 / THETA_S) / math.log1p(math.pi / THETA_S)
    assert math.pi - back > 0.09 * math.pi + mix * 0.4
