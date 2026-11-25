"""Cuboid targets: ``model_conf.cuboid_dim`` (edge lengths, only their ratios matter).

Positions are stored as the direction of ``x / cuboid_dim``, i.e. of the point on the
unit cube the cuboid maps to, so every face pair keeps a third of the sphere whatever
the aspect ratio; for a cube that is exactly the direction of ``x``. The slab below is
10 x 100 x 100 (half-sizes 5, 50, 50): a gun at (-5, 30, 0) firing along +x exits
at (5, 30, 0), whose stored direction is that of (1, 0.6, 0).
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import _F, _M
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from legofmt.config import resolve_legoltng_config
from legofmt.multiplicity.model import MultModel
from test_helpers import MANIFOLD, _flow_config, _gun_cond, _pt_mult_config, _save_mult_ckpt
from test_helpers import _head_batch as _batch, _head_config as _config

SLAB = [10.0, 100.0, 100.0]


def test_raytrace_exits_the_cuboid() -> None:
    x = torch.tensor([[-5.0, 30.0, 0.0], [-5.0, 30.0, 0.0]])
    p = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.1, 1.0, 0.0]]), dim=-1)
    out = CubeTrace(SLAB)(torch.cat((p, x), -1))[..., 3:]
    # straight through the thin axis; and out of the +y side face at x = -3
    assert torch.allclose(out, torch.tensor([[5.0, 30.0, 0.0], [-3.0, 50.0, 0.0]]), atol=1e-4)


def test_raytrace_cube_box_is_the_old_cube_trace() -> None:
    g = torch.Generator().manual_seed(0)
    x = F.normalize(torch.randn(256, 3, generator=g), dim=-1)
    x = 50.0 * x / x.abs().amax(-1, keepdim=True)
    p = F.normalize(torch.randn(256, 3, generator=g), dim=-1)
    ray = torch.cat((p, x), -1)
    assert torch.equal(CubeTrace([7.0, 7.0, 7.0])(ray), CubeTrace()(ray))


def test_an_exit_point_across_an_edge_is_mirrored_onto_the_face_the_momentum_leaves() -> None:
    """Exit points on the +x and -x faces just short of the edge to +y, momentum into that face and out through +y:
    the two largest |components| swap. An outward momentum, a point far from the edge, and a momentum into +y too
    stay."""
    u = F.normalize(torch.tensor([[1.0, 0.97, 0.2], [-1.0, 0.97, 0.2], [1.0, 0.97, 0.2], [1.0, 0.5, 0.2],
                                  [1.0, 0.97, 0.2]]), dim=-1)
    p = F.normalize(torch.tensor([[-0.3, 1.0, 0.0], [0.3, 1.0, 0.0], [0.3, 1.0, 0.0], [-0.3, 1.0, 0.0],
                                  [-0.3, -1.0, 0.0]]), dim=-1)
    want = u.clone()
    want[0] = F.normalize(torch.tensor([0.97, 1.0, 0.2]), dim=-1)
    want[1] = F.normalize(torch.tensor([-0.97, 1.0, 0.2]), dim=-1)
    assert torch.allclose(CubeTrace().exit_across_edge(p, u), want, atol=1e-6)


def _across_edge(p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Whether an exit point lies just across an edge from the face its momentum leaves through."""
    mag, ax = u.abs().sort(-1, descending=True)
    a, b = ax[..., :1], ax[..., 1:2]
    into = p.gather(-1, a) * u.gather(-1, a).sign() < 0
    out_b = p.gather(-1, b) * u.gather(-1, b).sign() > 0
    return (into & (mag[..., 1:2] > 0.95 * mag[..., :1]) & out_b).squeeze(-1)


def test_generate_leaves_no_exit_point_across_an_edge(tmp_path) -> None:
    """An untrained flow puts exit points and momenta at random, so some land across an edge; none is left."""
    cfg = _flow_config()
    torch.manual_seed(0)
    m = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": m.model.vf.state_dict(), "config": cfg}, tmp_path / "flow.pt")
    gen = GenerateOut(str(tmp_path / "flow.pt"), str(_save_mult_ckpt(tmp_path)), device="cpu")
    with torch.no_grad():
        sols, _, attn = gen(_gun_cond(22, batch=512))
    out = _F(sols)
    live = _M(attn).out_p
    assert live.sum() > 500
    assert not (_across_edge(out.out_mom, out.out_pos) & live).any()


def _prep(cuboid_dim=None) -> DataPrep:
    cfg = {"cutoff_mev": 10.0, "max_energy": 300.0, "manifold": MANIFOLD, "proj_ray": True}
    if cuboid_dim is not None:
        cfg["cuboid_dim"] = cuboid_dim
    return DataPrep(cfg)


def _slab_event() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(cc, e_kin)`` of one event: incoming at (-5, 30, 0) along +x, one outgoing particle exiting at
    (5, 40, 10)."""
    mom = torch.tensor([[[150.0, 0.0, 0.0], [0.0, 80.0, 0.0]]])
    pos = torch.tensor([[[-5.0, 30.0, 0.0], [5.0, 40.0, 10.0]]])
    return torch.cat((mom, pos), -1), torch.tensor([[[150.0], [80.0]]])


def test_prep_stores_positions_as_unit_box_directions() -> None:
    cc, e_kin = _slab_event()
    pos = _prep(SLAB).cc_trafo(cc, e_kin=e_kin)[..., 4:7]
    want = F.normalize(torch.tensor([[[1.0, 0.6, 0.0], [1.0, 0.8, 0.2]]]), dim=-1)
    assert torch.allclose(pos, want, atol=1e-5)


def test_prep_cube_box_matches_the_default() -> None:
    cc, e_kin = _slab_event()
    cc = cc.clone()
    cc[0, 0, 3:] = torch.tensor([-50.0, 30.0, 0.0])
    cc[0, 1, 3:] = torch.tensor([50.0, 40.0, 10.0])
    assert torch.equal(_prep([100.0] * 3).cc_trafo(cc, e_kin=e_kin), _prep().cc_trafo(cc, e_kin=e_kin))


def test_generate_preps_the_incoming_like_prep(tmp_path) -> None:
    cfg = _flow_config()
    cfg["model_conf"]["cuboid_dim"] = SLAB
    torch.manual_seed(0)
    m = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": m.model.vf.state_dict(), "config": cfg}, tmp_path / "flow.pt")
    gen = GenerateOut(str(tmp_path / "flow.pt"), str(_save_mult_ckpt(tmp_path)), device="cpu")
    cond = torch.tensor([[1.0, 150.0, 0.0, 0.0, -5.0, 30.0, 0.0, 22.0]])
    with torch.no_grad():
        sols, _, _ = gen(cond)
    want = F.normalize(torch.tensor([1.0, 0.6, 0.0]), dim=-1)
    assert torch.allclose(_F(sols).in_cc[0, 0, 4:7], want, atol=1e-5)


@pytest.mark.parametrize("cuboid_dim, chord", [(None, 1.0), (SLAB, 0.1)])
def test_base_head_chord_is_measured_in_the_cuboid(cuboid_dim, chord) -> None:
    """Incoming along +x: the chord is the x edge, in units of the longest edge."""
    cfg = _config(False)
    if cuboid_dim is not None:
        cfg["config"]["model_conf"]["cuboid_dim"] = cuboid_dim
    model = LEGOLtng(cfg)
    seen = []
    model.base_head[0].register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    model.base_head_params(_batch(torch.full((2,), 0.1), torch.tensor([1, 2])))
    assert torch.allclose(seen[0][:, -2], torch.full((2,), chord), atol=1e-6)


def test_canon_sym_rejects_a_cuboid() -> None:
    cfg = _flow_config()
    cfg["model_conf"].update(canon_sym=True, cuboid_dim=SLAB)
    with pytest.raises(ValueError, match="cuboid_dim"):
        LEGOLtng({"state_dict": {}, "config": cfg})


def test_cuboid_dim_comes_from_meta(tmp_path) -> None:
    meta = {"ntokens": 7, "particles": [22, 211, 2212], "max_energy": 300.0,
            "cutoff_mev": 10.0, "cond_scalars": ["Density"], "cuboid_dim": SLAB}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    cfg = _flow_config()
    cfg["dl_conf"]["lds_args"]["data"] = str(tmp_path)
    rc = resolve_legoltng_config({"config": cfg})
    assert rc.config["model_conf"]["cuboid_dim"] == SLAB


def test_to_cuboid_maps_stored_directions_back_onto_the_box() -> None:
    cc, e_kin = _slab_event()
    pos = _prep(SLAB).cc_trafo(cc, e_kin=e_kin)[..., 4:7]
    out = GeomTrafos().to_cuboid(torch.cat((torch.zeros_like(pos), pos), -1), d=torch.tensor(SLAB) / 2)[..., 3:]
    # the incoming particle is stored at the point where its ray leaves the cuboid, (5, 30, 0)
    assert torch.allclose(out, torch.tensor([[[5.0, 30.0, 0.0], [5.0, 40.0, 10.0]]]), atol=1e-4)


@pytest.mark.parametrize("pos_scale", [50.0, [5.0, 50.0, 50.0]])
def test_mult_pos_scale_is_the_box_half_width(pos_scale) -> None:
    """``mm_conf.pos_scale`` (a scalar, or one value per axis) is the ``d`` of ``to_cuboid``: a stored slab exit
    (1, 0.6, 0) enters the input projection at its physical point. It is not a ``state_dict`` key, so old checkpoints
    load as they are."""
    m = MultModel({"state_dict": {}, "config": _pt_mult_config(pos_scale=pos_scale)})
    d = torch.as_tensor(pos_scale)
    assert torch.equal(m.rc.pos_scale, d) and torch.equal(m.pos_scale, d) and "pos_scale" not in m.state_dict()
    x = torch.tensor([[0.5, 150.0, 1.0, 0.0, 0.0, *F.normalize(torch.tensor([1.0, 0.6, 0.0]), dim=-1)]])
    want = x.clone()
    want[:, -3:] = torch.tensor([1.0, 0.6, 0.0]) * d
    assert torch.allclose(m.proj_in(x), m.proj_in_(want), atol=1e-5)
