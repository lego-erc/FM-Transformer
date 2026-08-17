"""Smoke test for :class:`legofmt.distill.reflow.LEGOLtngDirect`.

Builds a tiny model from an in-memory config (no HF download, no
``meta.json``), constructs a synthetic batch matching the padded sequence
layout, and runs one forward + backward pass.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import torch

from legofmt.data.struct import DataStruct, _F
from legofmt.distill.reflow import LEGOLtngDirect as LEGOLtng
from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity


def _has_nonzero_grad(params) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


def _tiny_config() -> dict:
    pdgids = torch.tensor([22, 211, 2212], dtype=torch.int64).sort().values
    return {
        "state_dict": {},
        "config": {
            "dl_conf": {
                "lds_args": {"cutoff_mev": 10.0},
                "bs": 2,
                "num_workers": 0,
            },
            "val_conf": {"val_frac": 0.01, "seed": 0},
            "base_conf": {
                "base_range": 3.4,
                "kappa": torch.tensor(8.0),
                "bs_frac": 0.0,
                "base_dist": "poles",
                "scale_dist": "sm_norm",
                "tanh_theta": True,
            },
            "model_conf": {
                "manifold": [
                    {"name": "euclidean", "dim": 1},
                    {"name": "sphere",    "dim": 3},
                    {"name": "sphere",    "dim": 3},
                ],
                "max_energy": 300.0,
                "pdgids": pdgids,
                # base_pretrain_batches defaults to 300, which (with
                # scale_dist=sm_norm) builds base_head -- and that reads the
                # Z/A cond scalars this fixture does not carry.
                "base_pretrain_batches": 0,
                "model_args": {
                    "h_dim": 16,
                    "nlayers": 2,
                    "nhead": 2,
                    "in_dim": 7,
                    "max_seq_l": 5,
                    "ntypes": 4,
                    "nvtypes": 2,
                    "npdgids": pdgids.numel() + 1,
                    "ff_mult": 1,
                    "dropout": 0.0,
                    "use_adaptive_rmsnorm": True,
                    "use_adaptive_layerscale": True,
                    "ff_swish": True,
                    "ff_glu": True,
                },
            },
            "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
        },
    }


def _fake_batch(B: int = 2, L: int = 5) -> DataStruct:
    """Synthetic batch matching the padded sequence layout: 2 conditioning
    slots, 1 incoming-particle slot, then ``L - 3`` outgoing slots."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 0] = 0.1
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 2, 0] = 0.8
    f[:, 2, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 2, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f[:, 2, 7] = 22.0
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 0] = torch.rand(B, L - 3, generator=g)
    f[:, 3:, 1:4] = torch.nn.functional.normalize(
        torch.randn(B, L - 3, 3, generator=g), dim=-1,
    )
    f[:, 3:, 4:7] = torch.nn.functional.normalize(
        torch.randn(B, L - 3, 3, generator=g), dim=-1,
    )
    f[:, 3:, 7] = 211.0
    m = torch.zeros(B, L, dtype=torch.long)
    m[:, 3:] = 1
    am = torch.ones(B, L, dtype=torch.bool)
    return DataStruct(f, m, am)


def test_instantiates_and_steps() -> None:
    model = LEGOLtng(_tiny_config())
    model.on_fit_start()
    model.train()
    loss = model._step(_fake_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()
    assert _has_nonzero_grad(model.model.parameters()), "no nonzero gradients"


def test_forward_shapes_and_sphere_projection() -> None:
    model = LEGOLtng(_tiny_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch()
    out = model.model(
        model.gen_base_wrapper(ds),
        mask=ds.m.full, attn_mask=ds.am.full,
        types=model.types_embd, pdgids=model.convert_pdgids(ds.f.pdgids),
    )
    assert out.shape == ds.f.full.shape[:2] + (7,)
    dir_norms = out[:, 3:, 1:4].norm(dim=-1)
    pos_norms = out[:, 3:, 4:7].norm(dim=-1)
    assert torch.allclose(dir_norms, torch.ones_like(dir_norms), atol=1e-5), dir_norms
    assert torch.allclose(pos_norms, torch.ones_like(pos_norms), atol=1e-5), pos_norms


def _save_velocity_teacher(tmp_path) -> str:
    """Random-init velocity teacher saved to disk; reflow only needs ``solve`` to be callable."""
    cfg = _tiny_config()["config"]
    teacher = LEGOLtngVelocity({"state_dict": {}, "config": cfg})
    path = tmp_path / "velocity_teacher.pt"
    torch.save({"state_dict": teacher.model.vf.state_dict(), "config": cfg}, path)
    return str(path)


def test_reflow_uses_teacher_target(tmp_path) -> None:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["reflow_path"] = _save_velocity_teacher(tmp_path)
    cfg["config"]["model_conf"]["reflow_kwargs"] = {"method": "midpoint", "step_size": 0.5}

    model = LEGOLtng(cfg)
    assert model.reflow_teacher is not None
    model.on_fit_start(); model.train()

    loss = model._step(_fake_batch(), 0)
    assert torch.isfinite(loss), f"bad reflow loss: {loss}"
    loss.backward()
    assert _has_nonzero_grad(model.model.parameters()), "no student gradients"
    assert not _has_nonzero_grad(model.reflow_teacher.parameters()), "frozen teacher got grads"


def test_reflow_missing_path_warns_and_disables(tmp_path) -> None:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["reflow_path"] = str(tmp_path / "does_not_exist.pt")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LEGOLtng(cfg)
    assert model.reflow_teacher is None
    assert any("reflow_path" in str(w.message) for w in caught)


def test_solve_chunked_matches_full() -> None:
    """``split_size`` must yield bit-identical output, including for partial last chunks."""
    model = LEGOLtng(_tiny_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch(B=12)
    base = model.gen_base_wrapper(ds)
    full = model.solve(ds, x_init=base)
    for split in (12, 6, 5):
        chunked = model.solve(ds, x_init=base, split_size=split)
        assert torch.allclose(full, chunked, equal_nan=True), f"split_size={split} diverges"


def test_forward_honors_return_base() -> None:
    """``odeint_conf['return_base']`` short-circuits the model call."""
    model = LEGOLtng(_tiny_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch(B=4)
    object.__setattr__(model.rc, "odeint_conf", {"return_base": True})
    sols_base, _, _ = model(ds)
    object.__setattr__(model.rc, "odeint_conf", {})
    sols_real, _, _ = model(ds)
    assert not torch.allclose(sols_base, sols_real, equal_nan=True)


def test_reflow_teacher_isolated_from_submodule_registry(tmp_path) -> None:
    """The teacher must not be a registered submodule, otherwise Lightning's
    ``parent.train()`` flips ``teacher.model.training`` and invalidates the
    teacher's ``torch.compile`` training-mode guard on every train/eval
    transition (and DDP needlessly broadcasts its frozen weights)."""
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["reflow_path"] = _save_velocity_teacher(tmp_path)
    model = LEGOLtng(cfg)
    teacher = model.reflow_teacher
    assert teacher is not None

    assert not any(mod is teacher for mod in model.modules())

    teacher_first_param = next(teacher.parameters())
    assert id(teacher_first_param) not in {id(p) for p in model.parameters()}

    assert teacher.model.training is False
    model.train()
    assert teacher.model.training is False


def _step_cond_config(fac: float = 1.0, sections: int = 4) -> dict:
    cfg = _tiny_config()
    mc = cfg["config"]["model_conf"]
    mc["model_args"]["step_cond"] = True
    mc["one_step_euler_fac"] = fac
    mc["one_step_euler_sections"] = sections
    return cfg


def _vf_call(model, ds, **kw):
    return model.model(
        model.gen_base_wrapper(ds), mask=ds.m.full, attn_mask=ds.am.full,
        types=model.types_embd, pdgids=model.convert_pdgids(ds.f.pdgids), **kw,
    )


def _open_gate(model, v: float = 1.0):
    """step_gain is zero-init, so d has no effect until it is opened."""
    with torch.no_grad():
        model.model.vf.step_gain.fill_(v)
    return model


def test_step_cond_gain_is_zero_init_and_1d() -> None:
    """At init every d must reproduce the velocity slice: that is what stops an
    untrained step-conditioned slice from being arbitrarily wrong (it was, see
    the euler-vs-midpoint failure). 1-D so muon_factory routes it to AdamW --
    Newton-Schulz would discard the magnitude of a gate."""
    model = LEGOLtngVelocity(_step_cond_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch()
    assert model.model.vf.step_gain.ndim == 1
    assert float(model.model.vf.step_gain.detach().abs().sum()) == 0.0
    torch.manual_seed(0); ref = _vf_call(model, ds, t=torch.tensor(0.3))
    for dv in (0.0, 0.125, 0.5, 1.0):
        torch.manual_seed(0)
        out = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(dv))
        assert torch.equal(ref, out), f"d={dv} deviates from the velocity slice at init"


def test_step_cond_zero_is_the_velocity_slice() -> None:
    """With the gate open, d=0 must still reproduce the unconditioned field
    exactly -- that is what makes a velocity checkpoint a valid warm start."""
    model = _open_gate(LEGOLtngVelocity(_step_cond_config()))
    model.on_fit_start(); model.eval()
    ds = _fake_batch()
    torch.manual_seed(0); out_none = _vf_call(model, ds, t=torch.tensor(0.3))
    torch.manual_seed(0); out_zero = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(0.0))
    torch.manual_seed(0); out_half = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(0.5))
    assert torch.equal(out_none, out_zero), "d=0 differs from d=None"
    assert not torch.allclose(out_none, out_half), "gate open but d=0.5 changes nothing"


def test_step_cond_distinguishes_t_from_d() -> None:
    """Regression for 96344ee: emb_t(t) + emb_d(d) built on one frequency bank
    with the same sin/cos interleave is exactly symmetric under swapping t and
    d, so the model cannot tell (t=0.2, d=0.8) from (t=0.8, d=0.2)."""
    model = _open_gate(LEGOLtngVelocity(_step_cond_config()))
    model.on_fit_start(); model.eval()
    ds = _fake_batch()
    a = _vf_call(model, ds, t=torch.tensor(0.2), d=torch.tensor(0.8))
    b = _vf_call(model, ds, t=torch.tensor(0.8), d=torch.tensor(0.2))
    assert not torch.allclose(a, b), "(t, d) is degenerate under swapping"


def test_step_emb_is_smooth_in_d() -> None:
    """The property that actually matters and that I originally failed to check:
    nearby d must have SIMILAR embeddings, or the net cannot interpolate between
    the dyadic steps the loss trains and every d becomes an unrelated task. The
    old 8-octave bank scored 0.35 here. Also require ||emb_d|| -> 0 as d -> 0,
    since the average velocity over a vanishing interval is the instantaneous one."""
    vf = LEGOLtngVelocity(_step_cond_config()).model.vf
    emb = lambda d: (torch.tensor([[float(d)]]) * vf.freqs_d).sin()[0]
    cs = torch.nn.functional.cosine_similarity
    for d0 in (0.125, 0.25, 0.5, 1.0):
        sim = cs(emb(d0), emb(d0 + 0.01), dim=0)
        assert sim > 0.9, f"emb_d not smooth at d={d0}: cos={sim:.3f}"
    # still separable across octaves, otherwise d carries no information
    assert cs(emb(0.5), emb(0.25), dim=0) < 0.9, "octaves not distinguishable"
    # and it decays toward the velocity slice
    assert emb(2 ** -7).norm() < 0.4 * emb(0.5).norm()
    assert emb(0.0).norm() == 0.0


def test_flow_map_ladder_is_anchored_at_zero() -> None:
    """The bug behind the euler failure: predictions cover d in
    {2**0 .. 2**-(sections-1)} but the teacher queries d=step/2, which falls one
    octave below for the smallest step. That bottom rung was supervised by
    nothing, so the whole bootstrap stood on an arbitrary function."""
    from legofmt.distill.distill import one_step_euler_loss

    sections = 4
    model = LEGOLtngVelocity(_step_cond_config(sections=sections))
    model.on_fit_start(); model.train()
    ds = _fake_batch(B=64)
    base = model.gen_base_wrapper(ds)
    pid = model.convert_pdgids(ds.f.pdgids)

    supervised, teacher = set(), set()
    orig = model.model.vf.forward

    def spy(x, mask, attn_mask, types, pdgids_, t=None, d=None):
        out = orig(x, mask, attn_mask, types, pdgids_, t=t, d=d)
        if d is not None:
            vals = {round(float(v), 8) for v in d[mask == 1]}
            (supervised if torch.is_grad_enabled() else teacher).update(vals)
        return out

    model.model.vf.forward = spy
    for _ in range(300):
        one_step_euler_loss(model, base, ds, pid)
    model.model.vf.forward = orig

    assert 0.0 in teacher, "ladder is not anchored: d=0 is never the teacher"
    unsupervised = {d for d in teacher if d > 0} - supervised
    assert not unsupervised, f"teacher queries unsupervised d: {sorted(unsupervised)}"
    assert min(d for d in supervised) == 2.0 ** -(sections - 1)


def test_step_cond_flow_map_loss_steps() -> None:
    model = LEGOLtngVelocity(_step_cond_config())
    model.on_fit_start(); model.train()
    loss = model._step(_fake_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()
    assert _has_nonzero_grad(model.model.parameters()), "no nonzero gradients"


def test_one_step_euler_requires_step_cond() -> None:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["one_step_euler_fac"] = 1.0
    with pytest.raises(ValueError, match="step_cond"):
        LEGOLtngVelocity(cfg)


def test_one_step_euler_every_gates_the_term() -> None:
    """With every=2 an odd global_step must skip the term entirely."""
    cfg = _step_cond_config()
    cfg["config"]["model_conf"]["one_step_euler_every"] = 2
    model = LEGOLtngVelocity(cfg)
    model.on_fit_start(); model.train()
    seen = []
    orig = model.model.vf.forward
    model.model.vf.forward = lambda *a, _o=orig, **k: (seen.append(1), _o(*a, **k))[1]

    torch.manual_seed(0); model._step(_fake_batch(), 0)
    on = len(seen)                      # global_step 0: CFM + s1 + s2 + s_pred
    seen.clear()
    with patch.object(LEGOLtngVelocity, "global_step", 1):
        torch.manual_seed(0); model._step(_fake_batch(), 0)
    off = len(seen)                     # global_step 1: CFM only
    assert on == 4 and off == 1, f"expected 4 forwards on, 1 off; got {on}, {off}"


@pytest.mark.parametrize("gs", [0, 1])
def test_step_gain_gets_grad_even_when_flow_map_is_gated_off(gs) -> None:
    """step_gain is only reachable when d is not None, and the flow-map loss is
    the only caller that passes d. With one_step_euler_every > 1 the term is
    skipped on some steps, so step_gain would leave the graph and plain `ddp`
    (find_unused_parameters=False) raises. d=None must mean d=0, not skip."""
    cfg = _step_cond_config()
    cfg["config"]["model_conf"]["one_step_euler_every"] = 2
    model = LEGOLtngVelocity(cfg)
    model.on_fit_start(); model.train()
    with patch.object(LEGOLtngVelocity, "global_step", gs):
        model._step(_fake_batch(), 0).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"global_step={gs} left params without grad: {missing}"


def test_cond_init_scale_and_zero_biases() -> None:
    """The three source-indexed weights are summed then divided by 3, so each
    needs std sqrt(3)*Xavier for the effective map to be a proper in_dim->h_dim
    Xavier. xavier_normal_ on the 4-D tensors derived fan from dims 0/1 and came
    out 17x low, while the biases got Xavier-sized RANDOM noise that dominated
    the signal from x (~4x). Biases belong at zero."""
    vf = LEGOLtngVelocity(_tiny_config()).model.vf
    h, ind = vf.h_dim, vf.in_dim
    target = (2.0 / (ind + h)) ** 0.5
    for name in ("cond_w_mask", "cond_w_types", "cond_w_pdgids"):
        eff = getattr(vf, name).detach().std().item() * 3 ** 0.5 / 3
        assert abs(eff - target) / target < 0.25, f"{name}: effective std {eff:.4f} vs {target:.4f}"
    for name in ("cond_bi_mask", "cond_bi_types", "cond_bi_pdgids",
                 "cond_bo_mask", "cond_bo_types", "cond_bo_pdgids"):
        p = getattr(vf, name).detach()
        assert float(p.abs().sum()) == 0.0, f"{name} is not zero-init"


def test_flow_map_under_uncert_weighting() -> None:
    """The flow-map loss is ~1/2000 of the CFM loss at its raw scale, so it must
    be normalised by its own log-variance rather than a hand-set factor."""
    cfg = _step_cond_config()
    cfg["config"]["model_conf"]["uncert_weighting"] = True
    cfg["config"]["model_conf"]["uncert_bins"] = 8
    model = LEGOLtngVelocity(cfg)
    assert hasattr(model, "lv_flow") and model.lv_flow.ndim == 1
    ids = {id(p) for gr in model.opt.param_groups for p in gr["params"]}
    assert id(model.lv_flow) in ids, "lv_flow never reached the optimizer"
    model.on_fit_start(); model.train()
    model._step(_fake_batch(), 0).backward()
    assert model.lv_flow.grad is not None and float(model.lv_flow.grad.abs()) > 0

    # exp(lv_flow) must track E[L_flow]: drive it with a constant flow-map loss
    import legofmt.distill.distill as D
    const = 0.05
    with patch.object(D, "one_step_euler_loss", lambda *a, **k: torch.tensor(const)):
        with patch("legofmt.main.modules.one_step_euler_loss",
                   lambda *a, **k: torch.tensor(const)):
            opt = torch.optim.Adam([model.lv_flow], lr=0.1)
            for _ in range(400):
                opt.zero_grad()
                model._step(_fake_batch(), 0).backward()
                opt.step()
    got = float(model.lv_flow.detach().exp())
    assert abs(got - const) / const < 0.2, f"exp(lv_flow)={got:.4f}, expected ~{const}"


@pytest.mark.parametrize("gs", [0, 1])
def test_lv_flow_gets_grad_when_gated_off(gs) -> None:
    """Same DDP trap as step_gain: the barrier term must apply every step."""
    cfg = _step_cond_config()
    cfg["config"]["model_conf"]["uncert_weighting"] = True
    cfg["config"]["model_conf"]["one_step_euler_every"] = 2
    model = LEGOLtngVelocity(cfg)
    model.on_fit_start(); model.train()
    with patch.object(LEGOLtngVelocity, "global_step", gs):
        model._step(_fake_batch(), 0).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"global_step={gs} left params without grad: {missing}"


def _overflow_config(delta: float = 0.08) -> dict:
    cfg = _tiny_config()
    mc = cfg["config"]["model_conf"]
    mc["overflow_delta"] = delta
    return cfg


def _overflow_batch(B: int = 64) -> DataStruct:
    """Half the outgoing slots sit on the boundary peak (stored e == 0)."""
    ds = _fake_batch(B=B)
    f = ds.f.full.clone()
    f[: B // 2, 3:, 0] = 0.0
    f[B // 2:, 3:, 0] = 0.5
    f[:, 3:, 7] = 211.0
    return DataStruct(f, ds.m.full, ds.am.full)


def test_overflow_delta_zero_is_a_noop() -> None:
    plain = LEGOLtngVelocity(_tiny_config())
    off = LEGOLtngVelocity(_overflow_config(delta=0.0))
    off.model.load_state_dict(plain.model.state_dict())
    for m in (plain, off):
        m.on_fit_start(); m.train()
    ds = _overflow_batch()
    torch.manual_seed(0); a = plain._step(ds, 0)
    torch.manual_seed(0); b = off._step(ds, 0)
    assert torch.equal(a, b), f"{a.item()} != {b.item()}"


def test_overflow_target_shift_is_value_scoped() -> None:
    model = LEGOLtngVelocity(_overflow_config())
    ds = _overflow_batch()
    out = model._shift_overflow_targets(ds)
    e_new, e_old = _F(out.f.full).out_p[..., 0], _F(ds.f.full).out_p[..., 0]
    on_peak = e_old == 0
    assert torch.all(e_new[on_peak] == -0.08), "peak targets not shifted"
    assert torch.all(e_new[~on_peak] == e_old[~on_peak]), "off-peak targets moved"
    npf = model.rc.n_prefix
    assert torch.equal(out.f.full[:, :npf + 1], ds.f.full[:, :npf + 1]), "prefix/incoming touched"
    assert ds.f.full[0, 3, 0] == 0.0, "input batch was mutated in place"


def test_overflow_leaves_the_base_untouched() -> None:
    """overflow_delta must not perturb the base distribution: gen_base_wrapper
    output has to be identical with the shift on and off."""
    plain = LEGOLtngVelocity(_tiny_config())
    ov = LEGOLtngVelocity(_overflow_config())
    ov.model.load_state_dict(plain.model.state_dict())
    for m in (plain, ov):
        m.on_fit_start(); m.eval()
    ds = _overflow_batch(B=128)
    torch.manual_seed(0); b_plain = plain.gen_base_wrapper(ds)
    torch.manual_seed(0); b_ov = ov.gen_base_wrapper(ov._shift_overflow_targets(ds))
    assert torch.equal(b_plain, b_ov), "the base moved"
    assert torch.all(_F(b_ov).out_p[..., 0] > 0), "negative base energies appeared"


def test_overflow_negative_energy_decodes_to_the_peak() -> None:
    """to_mev clamps, so any overshoot below 0 lands exactly on the peak."""
    from legofmt.geometry.energy_proj import EnergyProjections

    pen = EnergyProjections(cutoff_mev=10.0, max_energy=300.0)
    e_in = torch.tensor([[0.9]])
    at_zero = pen.to_mev(torch.tensor([[0.0]]), e_in)
    for over in (-0.01, -0.08, -0.5):
        assert torch.equal(pen.to_mev(torch.tensor([[over]]), e_in), at_zero)


def test_step_cond_euler_step_budgets_differ() -> None:
    model = LEGOLtngVelocity(_step_cond_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch(B=4)
    base = model.gen_base_wrapper(ds)
    one = model.solve(ds, x_init=base, method="euler", step_size=1.0)
    four = model.solve(ds, x_init=base, method="euler", step_size=0.25)
    assert one.shape == four.shape == base.shape
    assert torch.isfinite(one).all() and torch.isfinite(four).all()
    assert not torch.allclose(one, four), "step budget had no effect"


def test_step_cond_loads_legacy_state_dict() -> None:
    """Loading an older checkpoint must leave step_gain at 0, which makes the
    model ignore d entirely -- so a checkpoint trained against the old wide bank
    degrades to its (good) velocity field rather than to a broken flow map.
    freqs_d is non-persistent, so it never travels in or out of a state_dict."""
    plain = LEGOLtngVelocity(_tiny_config())
    sd = plain.model.vf.state_dict()
    model = LEGOLtngVelocity(_step_cond_config())
    bank = model.model.vf.freqs_d.clone()

    assert "freqs_d" not in sd
    assert "freqs_d" not in model.model.vf.state_dict(), "freqs_d must not be persistent"

    res = model.model.vf.load_state_dict(sd, strict=False)
    assert res.missing_keys == ["step_gain"], res.missing_keys
    assert not res.unexpected_keys, res.unexpected_keys
    assert torch.equal(model.model.vf.freqs_d, bank)
    assert float(model.model.vf.step_gain.detach().abs().sum()) == 0.0

    # an OLD step_cond ckpt carried freqs_d as a persistent buffer: tolerated
    stale = {**sd, "freqs_d": torch.zeros_like(bank)}
    res = model.model.vf.load_state_dict(stale, strict=False)
    assert res.unexpected_keys == ["freqs_d"], res.unexpected_keys
    assert torch.equal(model.model.vf.freqs_d, bank), "stale bank overwrote the new one"


def _uncert_config(bins: int = 8) -> dict:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["uncert_weighting"] = True
    cfg["config"]["model_conf"]["uncert_bins"] = bins
    return cfg


def test_uncert_zero_init_matches_unweighted() -> None:
    """lv=0 -> weight exp(0)=1 and a zero barrier, so switching the knob on
    cannot change where training starts. Not bit-exact: the weighted branch
    multiplies before reducing, which reorders the float32 summation."""
    plain = LEGOLtngVelocity(_tiny_config())
    unc = LEGOLtngVelocity(_uncert_config())
    unc.model.load_state_dict(plain.model.state_dict())
    for m in (plain, unc):
        m.on_fit_start(); m.train()
    ds = _fake_batch()
    torch.manual_seed(0); l_plain = plain._step(ds, 0)
    torch.manual_seed(0); l_unc = unc._step(ds, 0)
    assert torch.allclose(l_plain, l_unc, rtol=1e-6, atol=0), \
        f"{l_plain.item()} != {l_unc.item()}"


def test_uncert_param_is_1d_so_muon_skips_it() -> None:
    """muon_factory routes ndim>=2 to Muon, whose Newton-Schulz step discards
    magnitude -- fatal for a log-variance. Keep lv 1-D."""
    from legofmt.mod_comps.optimizers import muon_factory

    model = LEGOLtngVelocity(_uncert_config(bins=8))
    assert model.lv.ndim == 1 and model.lv.numel() == 8 * 3
    # the tiny config uses schedulefree, so build a Muon the way opt: muon would
    opt = muon_factory([*model.model.parameters(), model.lv], lr=1e-3)
    muon = [gr for gr in opt.param_groups if gr.get("use_muon")]
    adamw = [gr for gr in opt.param_groups if not gr.get("use_muon")]
    assert muon and adamw, "expected both a Muon and an AdamW group"
    assert not any(p is model.lv for gr in muon for p in gr["params"]), "lv went to Muon"
    assert any(p is model.lv for gr in adamw for p in gr["params"]), "lv reached no group"


def test_uncert_lv_receives_grad_and_tracks_loss_scale() -> None:
    """The barrier fixes exp(lv) -> E[L|t], so a block with a larger loss must
    end up with a larger fitted variance."""
    model = LEGOLtngVelocity(_uncert_config(bins=1))  # one bin: pure scale fit
    model.on_fit_start(); model.train()
    ds = _fake_batch(B=8)
    opt = torch.optim.Adam([model.lv], lr=0.2)
    scales = None
    for _ in range(150):
        opt.zero_grad()
        sq = torch.zeros(8, 5, 7)
        sq[..., 0:1] = 4.0      # energy block: large loss
        sq[..., 1:4] = 1.0      # dir block:    medium
        sq[..., 4:7] = 0.25     # pos block:    small
        loss = model._reduce_and_log(sq, ds, 0.0, t=torch.full((8,), 0.5))
        loss.backward()
        opt.step()
        scales = model.lv.detach().exp()
    assert model.lv.grad is not None and model.lv.grad.abs().sum() > 0
    assert scales[0] > scales[1] > scales[2], f"variances not ordered: {scales}"
    # exp(lv) should land on the per-block mean squared error itself
    assert torch.allclose(scales, torch.tensor([4.0, 1.0, 0.25]), rtol=0.1), scales


def test_uncert_floor_caps_the_weight() -> None:
    """A near-zero loss would send w=exp(-lv) to infinity; uncert_min bounds it."""
    cfg = _uncert_config(bins=1)
    cfg["config"]["model_conf"]["uncert_min"] = -2.0
    model = LEGOLtngVelocity(cfg)
    model.on_fit_start(); model.train()
    ds = _fake_batch(B=8)
    opt = torch.optim.Adam([model.lv], lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        loss = model._reduce_and_log(
            torch.full((8, 5, 7), 1e-8), ds, 0.0, t=torch.full((8,), 0.5))
        loss.backward()
        opt.step()
    used = model.lv.detach().clamp(min=-2.0)
    assert torch.all(used >= -2.0 - 1e-6)
    assert torch.exp(-used).max() <= torch.exp(torch.tensor(2.0)) + 1e-4


def test_uncert_rejects_time_independent_path() -> None:
    """LEGOLtngDirect has no per-event t, so weighting must fail loudly."""
    model = LEGOLtng(_uncert_config())          # LEGOLtng here is LEGOLtngDirect
    model.on_fit_start(); model.train()
    with pytest.raises(ValueError, match="per-event t"):
        model._step(_fake_batch(), 0)


@pytest.mark.parametrize("cls", [LEGOLtngVelocity, LEGOLtng])
def test_all_params_receive_grad(cls) -> None:
    """One training step touches every parameter (DDP-safety with
    ``find_unused_parameters=False``)."""
    model = cls(_tiny_config())
    model.on_fit_start()
    model.train()
    loss = model._step(_fake_batch(), 0)
    loss.backward()
    missing = [n for n, p in model.model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad (DDP would crash): {missing}"
