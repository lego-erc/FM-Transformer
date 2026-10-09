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
from test_generate_direct import _save_flow_ckpt
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
    assert LEGOLtng(cfg) is not None
