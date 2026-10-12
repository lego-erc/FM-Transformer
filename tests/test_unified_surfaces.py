"""Every pre-unification surface still works after the species channel landed.

Written after four regressions survived a green suite: ``couple_in_out_pdgids``
raised at construction, ``GenerateOutDirect`` produced empty events,
``return_timesteps`` broke on the pass-through short-circuit, and ``gt_mult``
was ignored because a pinned-empty slot is indistinguishable from an
unassigned one unless its mask is switched off. Each of those is a test here.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import _F, set_layout
from legofmt.distill.reflow import GenerateOutDirect
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_generate_direct import _flow_config, _save_flow_ckpt
from test_modules_direct import _fake_batch, _tiny_config


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory: pytest.TempPathFactory) -> str:
    torch.manual_seed(0)
    return str(_save_flow_ckpt(tmp_path_factory.mktemp("surfaces")))


@pytest.fixture(scope="module")
def gen(ckpts) -> GenerateOut:
    g = GenerateOut(ckpts, device="cpu")
    with torch.no_grad():               # suppress the gate: exercise the flow path
        g.model.pt_head[-1].bias.fill_(-50.0)
        g.model.pt_head[-1].weight.zero_()
    return g


def _cond(g, n: int = 4) -> torch.Tensor:
    set_layout(g.cond_names)
    c = torch.zeros(n, g.n_cond + 7)
    c[:, 0] = 1.0
    c[:, g.n_cond:g.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    c[:, g.n_cond + 3:g.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    c[:, -1] = g.pdgids[0].item()
    return c


@pytest.mark.parametrize("key,value,check", [
    ("return_base", True, lambda s: s.shape[-1] == 8),
    ("return_timesteps", True, lambda s: s.dim() == 4),
    ("split_size", 2, lambda s: s.dim() == 3),
    ("compile_buckets", [8], lambda s: s.dim() == 3),
    ("method", "euler", lambda s: s.dim() == 3),
    ("method", "midpoint", lambda s: s.dim() == 3),
    ("method", "rk4", lambda s: s.dim() == 3),
])
def test_odeint_conf_surfaces(gen: GenerateOut, key, value, check) -> None:
    cfg = gen.model.rc.odeint_conf
    prev, had = cfg.get(key), key in cfg
    cfg[key] = value
    try:
        sols, _, _ = gen(_cond(gen, 5))
    finally:
        cfg[key] = prev if had else cfg.pop(key, None)
        if not had:
            cfg.pop(key, None)
    assert check(sols), f"{key}={value} produced {sols.shape}"
    if key == "method":                                  # the species channel must actually be generated
        assert (sols[:, gen.n_prefix + 1:, -1].nan_to_num(0.0) != 0).any(), f"{value} generated no particle"


def test_methods_without_species_jumps_raise(gen: GenerateOut) -> None:
    """The ODESolver methods (adaptive dopri5, ...) never jump the species, so a model that
    generates its species channel would emit only the pinned slots (rk4 did: n_out 0.46 on
    count_s6, before it got its own stepper)."""
    cfg = gen.model.rc.odeint_conf
    prev, had = cfg.get("method"), "method" in cfg
    cfg["method"] = "dopri5"
    try:
        with pytest.raises(ValueError, match="species"):
            gen(_cond(gen, 3))
    finally:
        if had:
            cfg["method"] = prev
        else:
            cfg.pop("method", None)


def test_heads_are_queried_in_the_canonical_frame(tmp_path) -> None:
    """With canon_sym the gate, survival and count heads are trained after _step canonicalises
    the batch, so generation has to hand them the canonical frame too: three guns on three
    faces must reach _gate_x on one face."""
    cfg = _flow_config()
    cfg["model_conf"]["canon_sym"] = True
    m = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": m.model.vf.state_dict(), "config": cfg}, tmp_path / "f.pt")
    gen = GenerateOut(str(tmp_path / "f.pt"), device="cpu")
    cond = _cond(gen, 3)                                   # -z face, momentum +z
    nc = gen.n_cond
    cond[1, nc:nc + 3], cond[1, nc + 3:nc + 6] = torch.tensor([150.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 0.0])
    cond[2, nc:nc + 3], cond[2, nc + 3:nc + 6] = torch.tensor([0.0, 150.0, 0.0]), torch.tensor([0.0, -1.0, 0.0])
    assert gen.model.sym.face_of(cond[:, nc + 3:nc + 6]).unique().numel() == 3
    seen, orig = [], gen.model._gate_x
    def spy(ds_t, idx=None):
        seen.append(ds_t.f.in_cc[..., 0, 4:7].clone())
        return orig(ds_t, idx)
    gen.model._gate_x = spy
    gen(cond)
    assert seen, "no head was queried"
    for pos in seen:
        faces = gen.model.sym.face_of(pos)
        assert faces.unique().numel() == 1, f"a head saw the raw frame: faces {faces.tolist()}"


def test_passthrough_shortcircuit_survives_return_timesteps(gen: GenerateOut) -> None:
    """Fired rows are written before the (T, B, L, C) expand, so they must come out
    constant in time while the solved rows evolve.

    A mixed batch is the case that matters, and the gate makes it deterministic:
    it is masked to neutrals, so photons fire and protons never do.
    """
    cond = _cond(gen, 4)
    cond[2:, -1] = 2212                       # charged -> always routed through the flow
    with torch.no_grad():
        gen.model.pt_head[-1].bias.fill_(50.0)
    gen.model.rc.odeint_conf["return_timesteps"] = True
    try:
        sols, _, _ = gen(cond)
    finally:
        gen.model.rc.odeint_conf.pop("return_timesteps")
        with torch.no_grad():
            gen.model.pt_head[-1].bias.fill_(-50.0)

    assert sols.dim() == 4 and sols.shape[0] > 1, sols.shape
    # torch.equal is False on NaN, and the pad slots are NaN by construction
    fired = sols[:, :2].nan_to_num(-1.0)
    assert torch.equal(fired[0], fired[-1]), "a fired row drifted across timesteps"
    flowed = sols[:, 2:].nan_to_num(-1.0)
    assert not torch.equal(flowed[0], flowed[-1]), "the solved rows never moved"


def test_couple_in_out_pdgids_constructs(ckpts) -> None:
    g = GenerateOut(ckpts, device="cpu", couple_in_out_pdgids=True)
    assert g.model.rc.odeint_conf["filter_pdgid"] is not None


def test_gt_mult_pins_the_multiplicity(ckpts) -> None:
    g = GenerateOut(ckpts, device="cpu")
    gt = torch.zeros(4, g.pdgids.shape[0], dtype=torch.long)
    gt[:, 0] = 2
    _, _, attn = g(_cond(g, 4), gt_mult=gt)
    assert (attn[:, g.n_prefix + 1:].sum(-1) == 2).all()


def test_direct_generator_still_emits_particles(ckpts) -> None:
    """The direct model cannot jump the class along a trajectory, but it predicts
    it in one shot -- without that it produced entirely empty events."""
    gd = GenerateOutDirect(ckpts, device="cpu")
    with torch.no_grad():
        gd.model.pt_head[-1].bias.fill_(-50.0)
        gd.model.pt_head[-1].weight.zero_()
    torch.manual_seed(0)
    sols, _, attn = gd(_cond(gd, 6))
    assert attn[:, gd.n_prefix + 1:].sum() > 0, "direct generator produced nothing"
    assert not _F(sols).out_p[..., :7].isnan().all()


def test_log_likelihood_and_reverse_solve_still_run() -> None:
    m = LEGOLtng(_tiny_config())
    m.on_fit_start(); m.eval()
    ds = _fake_batch()
    assert m.log_likelihood(ds, lambda x: torch.zeros(x.shape[0]), step_size=0.5).shape[0] == 2
    assert m.solve(ds, reverse=True, step_size=0.5).shape == ds.f.model_in.shape


def test_exclude_passthrough_config_still_builds() -> None:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["exclude_passthrough"] = True
    with pytest.raises(ValueError, match="pass-through gate"):   # the gate would train without a positive
        LEGOLtng(cfg)
    cfg["config"]["model_conf"]["passthrough_fac"] = 0.0
    assert LEGOLtng(cfg) is not None


@pytest.mark.parametrize("timesteps", [False, True])
def test_occupied_slots_come_first(gen: GenerateOut, timesteps: bool) -> None:
    """The species channel can leave any candidate slot empty, but consumers slice by slot
    (CornerHist cuts the model output to the truth's length), so the generator must pack the
    occupied outgoing slots at the front, as the data and the two-model generator do. The
    scattered layout dropped half of the particles from the stack corner plot."""
    torch.manual_seed(0)
    n0, orig = gen.n_prefix + 1, gen.model.forward

    def gappy(batch, *a, **k):                    # the first candidate slot resolves empty
        sols, m, am = orig(batch, *a, **k)
        am, sols = am.clone(), sols.clone()
        am[:, n0] = False
        sols[..., n0, :7] = torch.nan
        return sols, m, am

    gen.model.forward = gappy
    if timesteps:
        gen.model.rc.odeint_conf["return_timesteps"] = True
    try:
        sols, _, attn = gen(_cond(gen, 64))
    finally:
        gen.model.rc.odeint_conf.pop("return_timesteps", None)
        del gen.model.forward
    occ = attn[:, gen.n_prefix + 1:]
    assert occ.any(-1).any() and (~occ).any(), "fixture no longer mixes occupied and empty slots"
    assert not (~occ[:, :-1] & occ[:, 1:]).any(), "an occupied slot follows an empty one"
    out = _F(sols[-1] if timesteps else sols).out_p
    assert out[occ][:, :7].isfinite().all() and out[~occ][:, :7].isnan().all()
