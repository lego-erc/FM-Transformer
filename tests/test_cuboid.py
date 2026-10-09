"""Cuboid targets: ``model_conf.cuboid_dim`` (edge lengths, only their ratios matter).

Positions are stored as the direction of ``x / cuboid_dim``, i.e. of the point on the
unit cube the cuboid maps to, so every face pair keeps a third of the sphere whatever
the aspect ratio; for a cube that is exactly the old ``dir(x)``. The slab below is
10 x 100 x 100 (half-sizes 5, 50, 50): a gun at (-5, 30, 0) firing along +x exits
at (5, 30, 0), whose stored direction is that of (1, 0.6, 0).
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from legofmt.mod_comps.config import resolve_legoltng_config
from test_base_head_nout import _batch, _config
from test_generate_direct import _flow_config

SLAB = [10.0, 100.0, 100.0]
MANIFOLD = [{"name": "euclidean", "dim": 1},
            {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}]


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


def _prep(cuboid_dim=None) -> DataPrep:
    cfg = {"cutoff_mev": 10.0, "max_energy": 300.0, "manifold": MANIFOLD, "proj_ray": True}
    if cuboid_dim is not None:
        cfg["cuboid_dim"] = cuboid_dim
    return DataPrep(cfg)


def _slab_event() -> tuple[torch.Tensor, torch.Tensor]:
    """Incoming at (-5, 30, 0) along +x; one outgoing exiting at (5, 40, 10)."""
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
    gen = GenerateOut(str(tmp_path / "flow.pt"), device="cpu")
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
