"""Tests for the training side of tiny in-memory flows: :class:`legofmt.legacy.reflow.LEGOLtngDirect` (imported here
as ``LEGOLtng``) and :class:`legofmt.main.modules.LEGOLtng` (here ``LEGOLtngVelocity``). They cover reflow, the flow
map (``step_cond``), ``overflow_delta`` and the learned loss weights."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_helpers import _fake_batch, _tiny_config, _vf_kwargs
from legofmt.data.struct import DataStruct, _F
from legofmt.legacy.reflow import LEGOLtngDirect as LEGOLtng
from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity


def _has_nonzero_grad(params) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


def _fit(model, train: bool = True):
    """Returns the model as Lightning hands it to a step: after ``on_fit_start``, in train or eval mode."""
    model.on_fit_start()
    return model.train(train)


def _step_cond_config(**model_conf) -> dict:
    """Returns the config of a flow-map field (``step_cond``) with the flow-map term on: factor 1, 4 sections."""
    return _tiny_config({"step_cond": True}, **{"one_step_euler_fac": 1.0, "one_step_euler_sections": 4, **model_conf})


def _twins(cfg: dict, train: bool):
    """Returns a plain tiny flow and one built from ``cfg``, both with the same weights."""
    plain, other = LEGOLtngVelocity(_tiny_config()), LEGOLtngVelocity(cfg)
    other.model.load_state_dict(plain.model.state_dict())
    return _fit(plain, train), _fit(other, train)


@pytest.mark.parametrize("cls, cfg", [(LEGOLtng, _tiny_config), (LEGOLtngVelocity, _tiny_config),
                                      (LEGOLtngVelocity, _step_cond_config)], ids=["direct", "velocity", "flow_map"])
def test_training_step_gives_every_param_a_grad(cls, cfg) -> None:
    """The loss is a finite scalar and its backward reaches every parameter: DDP with
    ``find_unused_parameters=False`` raises on a parameter without a gradient."""
    model = _fit(cls(cfg()))
    loss = model._step(_fake_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()
    assert _has_nonzero_grad(model.model.parameters()), "no nonzero gradients"
    missing = [n for n, p in model.model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad (DDP would crash): {missing}"


def test_forward_shapes_and_sphere_projection() -> None:
    model = _fit(LEGOLtng(_tiny_config()), train=False)
    ds = _fake_batch()
    out = model.model(model.gen_base_wrapper(ds), **_vf_kwargs(model, ds))
    assert out.shape == ds.f.full.shape[:2] + (7,)
    dir_norms = out[:, 3:, 1:4].norm(dim=-1)
    pos_norms = out[:, 3:, 4:7].norm(dim=-1)
    assert torch.allclose(dir_norms, torch.ones_like(dir_norms), atol=1e-5), dir_norms
    assert torch.allclose(pos_norms, torch.ones_like(pos_norms), atol=1e-5), pos_norms


def _save_velocity_teacher(tmp_path) -> str:
    """Saves a randomly initialised teacher flow and returns its path; reflow only needs its ``solve`` to run."""
    cfg = _tiny_config()["config"]
    teacher = LEGOLtngVelocity({"state_dict": {}, "config": cfg})
    path = tmp_path / "velocity_teacher.pt"
    torch.save({"state_dict": teacher.model.vf.state_dict(), "config": cfg}, path)
    return str(path)


def test_reflow_uses_teacher_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda m, **k: m)  # inductor would cost ~10 s
    model = LEGOLtng(_tiny_config(reflow_path=_save_velocity_teacher(tmp_path),
                                  reflow_kwargs={"method": "midpoint", "step_size": 0.5}))
    assert model.reflow_teacher is not None
    _fit(model)
    loss = model._step(_fake_batch(), 0)
    assert torch.isfinite(loss), f"bad reflow loss: {loss}"
    loss.backward()
    assert _has_nonzero_grad(model.model.parameters()), "no student gradients"
    assert not _has_nonzero_grad(model.reflow_teacher.parameters()), "frozen teacher got grads"


def test_reflow_missing_path_warns_and_disables(tmp_path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LEGOLtng(_tiny_config(reflow_path=str(tmp_path / "does_not_exist.pt")))
    assert model.reflow_teacher is None
    assert any("reflow_path" in str(w.message) for w in caught)


def test_solve_chunked_matches_full() -> None:
    """``split_size`` must not change the solve's output, also when the last chunk is a partial one."""
    model = _fit(LEGOLtng(_tiny_config()), train=False)
    ds = _fake_batch(B=12)
    base = model.gen_base_wrapper(ds)
    full = model.solve(ds, x_init=base)
    for split in (12, 6, 5):
        chunked = model.solve(ds, x_init=base, split_size=split)
        assert torch.allclose(full, chunked, equal_nan=True), f"split_size={split} diverges"


def test_forward_honors_return_base() -> None:
    """With ``odeint_conf.return_base``, ``forward`` returns the base itself instead of a solve from it."""
    model = _fit(LEGOLtng(_tiny_config()), train=False)
    ds = _fake_batch(B=4)
    object.__setattr__(model.rc, "odeint_conf", {"return_base": True})
    sols_base, _, _ = model(ds)
    object.__setattr__(model.rc, "odeint_conf", {})
    sols_real, _, _ = model(ds)
    assert not torch.allclose(sols_base, sols_real, equal_nan=True)


def test_reflow_teacher_isolated_from_submodule_registry(tmp_path) -> None:
    """The teacher must not be a registered submodule: Lightning's ``parent.train()`` would flip
    ``teacher.model.training`` and invalidate the teacher's ``torch.compile`` training-mode guard on every train/eval
    transition, and DDP would broadcast its frozen weights."""
    model = LEGOLtng(_tiny_config(reflow_path=_save_velocity_teacher(tmp_path)))
    teacher = model.reflow_teacher
    assert teacher is not None

    assert not any(mod is teacher for mod in model.modules())

    teacher_first_param = next(teacher.parameters())
    assert id(teacher_first_param) not in {id(p) for p in model.parameters()}

    assert teacher.model.training is False
    model.train()
    assert teacher.model.training is False


def _vf_call(model, ds, **kw):
    return model.model(model.gen_base_wrapper(ds), **_vf_kwargs(model, ds), **kw)


def _open_gate(model, v: float = 1.0):
    """Sets ``step_gain`` to ``v``; at its initial 0, ``d`` has no effect on the field."""
    with torch.no_grad():
        model.model.vf.step_gain.fill_(v)
    return model


def test_step_cond_gain_is_zero_init_and_1d() -> None:
    """At initialisation the field must give the instantaneous velocity for every ``d``: otherwise an untrained flow
    map is arbitrarily wrong at ``d > 0``. ``step_gain`` is 1-D so that ``muon_factory`` routes it to AdamW:
    Newton-Schulz would discard the magnitude of a gain."""
    model = _fit(LEGOLtngVelocity(_step_cond_config()), train=False)
    ds = _fake_batch()
    assert model.model.vf.step_gain.ndim == 1
    assert float(model.model.vf.step_gain.detach().abs().sum()) == 0.0
    torch.manual_seed(0); ref = _vf_call(model, ds, t=torch.tensor(0.3))
    for dv in (0.0, 0.125, 0.5, 1.0):
        torch.manual_seed(0)
        out = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(dv))
        assert torch.equal(ref, out), f"d={dv} deviates from the velocity slice at init"


def test_step_cond_zero_is_the_velocity_slice() -> None:
    """With a non-zero ``step_gain``, ``d = 0`` must still reproduce the field without ``d`` exactly: that makes a
    checkpoint trained without the flow map a valid warm start."""
    model = _fit(_open_gate(LEGOLtngVelocity(_step_cond_config())), train=False)
    ds = _fake_batch()
    torch.manual_seed(0); out_none = _vf_call(model, ds, t=torch.tensor(0.3))
    torch.manual_seed(0); out_zero = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(0.0))
    torch.manual_seed(0); out_half = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(0.5))
    assert torch.equal(out_none, out_zero), "d=0 differs from d=None"
    assert not torch.allclose(out_none, out_half), "gate open but d=0.5 changes nothing"


def test_step_cond_distinguishes_t_from_d() -> None:
    """The embeddings of ``t`` and ``d`` must differ in form: built on one frequency bank with the same sin/cos
    interleave, their sum is symmetric under swapping ``t`` and ``d``, and the field cannot tell (t=0.2, d=0.8) from
    (t=0.8, d=0.2)."""
    model = _fit(_open_gate(LEGOLtngVelocity(_step_cond_config())), train=False)
    ds = _fake_batch()
    a = _vf_call(model, ds, t=torch.tensor(0.2), d=torch.tensor(0.8))
    b = _vf_call(model, ds, t=torch.tensor(0.8), d=torch.tensor(0.2))
    assert not torch.allclose(a, b), "(t, d) is degenerate under swapping"


def test_step_emb_is_smooth_in_d() -> None:
    """Nearby ``d`` must have similar embeddings, or the field cannot interpolate between the dyadic steps the loss
    trains and every ``d`` is an unrelated task (an 8-octave frequency bank has cosine similarity 0.35 here). The
    embedding must also vanish as ``d -> 0``: the average velocity over a vanishing interval is the instantaneous
    one."""
    vf = LEGOLtngVelocity(_step_cond_config()).model.vf
    emb = lambda d: (torch.tensor([[float(d)]]) * vf.freqs_d).sin()[0]
    cs = torch.nn.functional.cosine_similarity
    for d0 in (0.125, 0.25, 0.5, 1.0):
        sim = cs(emb(d0), emb(d0 + 0.01), dim=0)
        assert sim > 0.9, f"emb_d not smooth at d={d0}: cos={sim:.3f}"
    # still separable across octaves, otherwise d carries no information
    assert cs(emb(0.5), emb(0.25), dim=0) < 0.9, "octaves not distinguishable"
    # and it decays towards d = 0, the instantaneous velocity
    assert emb(2 ** -7).norm() < 0.4 * emb(0.5).norm()
    assert emb(0.0).norm() == 0.0


def test_flow_map_ladder_is_anchored_at_zero() -> None:
    """The predictions cover ``d`` in {2**0, ..., 2**-(sections - 1)}, but the teacher asks for ``d / 2``, which for
    the smallest step is one octave below them and supervised by nothing. The smallest step's teacher must therefore
    use ``d = 0``, the instantaneous velocity: otherwise the whole ladder stands on an arbitrary function."""
    from legofmt.cfm.flow_map import one_step_euler_loss

    sections = 4
    model = _fit(LEGOLtngVelocity(_step_cond_config(one_step_euler_sections=sections)))
    ds = _fake_batch(B=64)
    base = model.gen_base_wrapper(ds)
    pid = model.convert_pdgids(ds.f.pdgids)

    supervised, teacher = set(), set()
    orig = model.model.vf.forward

    def spy(x, mask, attn_mask, types, pdgids_, t=None, d=None):
        out = orig(x, mask, attn_mask, types, pdgids_, t=t, d=d)
        if d is not None:
            vals = {round(float(v), 8) for v in d.expand_as(mask)[mask == 1]}
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


# ---- the rung curriculum (one_step_euler_unlock) -------------------------------------------------------------------

def _at_fraction(f: float, total: int = 100):
    """Puts the training clock at ``f`` of a run of ``total`` optimizer steps (``global_step``, the trainer's count)."""
    return patch.multiple(LEGOLtngVelocity, global_step=round(f * total),
                          trainer=SimpleNamespace(estimated_stepping_batches=total))


def _steps_asked(model, calls: int = 40) -> tuple[set, set]:
    """Returns the step sizes ``d`` that ``calls`` flow-map losses ask of the field: (predictions, teacher half steps)."""
    from legofmt.cfm.flow_map import one_step_euler_loss

    ds = _fake_batch(B=64)
    base, pid = model.gen_base_wrapper(ds), model.convert_pdgids(ds.f.pdgids)
    predicted, teacher = set(), set()
    orig = model.model.vf.forward

    def spy(x, mask, attn_mask, types, pdgids_, t=None, d=None):
        if d is not None:
            vals = {round(float(v), 8) for v in d.expand_as(mask)[mask == 1]}
            (predicted if torch.is_grad_enabled() else teacher).update(vals)
        return orig(x, mask, attn_mask, types, pdgids_, t=t, d=d)

    model.model.vf.forward = spy
    for _ in range(calls):
        one_step_euler_loss(model, base, ds, pid)
    model.model.vf.forward = orig
    return predicted, teacher


def test_one_step_euler_unlock_resolves() -> None:
    """The key must reach the run config: an unknown ``model_conf`` key is dropped without a word, which would train a
    curriculum run on every rung from step 0."""
    assert LEGOLtngVelocity(_step_cond_config()).rc.one_step_euler_unlock == 0.0
    assert LEGOLtngVelocity(_step_cond_config(one_step_euler_unlock=0.5)).rc.one_step_euler_unlock == 0.5
    for u in (-0.1, 1.5):
        with pytest.raises(ValueError, match="one_step_euler_unlock"):
            LEGOLtngVelocity(_step_cond_config(one_step_euler_unlock=u))


@pytest.mark.parametrize("f, steps", [(0.0, {0.125}), (0.2, {0.25, 0.125}), (0.5, {1.0, 0.5, 0.25, 0.125})])
def test_curriculum_unlocks_the_rungs_from_the_smallest_step_up(f, steps) -> None:
    """With u = 0.5 and 4 sections, n = 1 + floor(3 min(1, f / 0.5)) rungs are drawn, the smallest steps first: at
    f = 0 only d = 1/8, whose teacher stays the instantaneous velocity (d = 0); every rung from f = u on."""
    model = _fit(LEGOLtngVelocity(_step_cond_config(one_step_euler_unlock=0.5)))
    with _at_fraction(f):
        predicted, teacher = _steps_asked(model)
    assert predicted == steps
    assert 0.0 in teacher and teacher - {0.0} <= predicted


def test_validation_draws_every_rung() -> None:
    """Validation scores every rung at any point of training, so its flow-map loss compares across runs."""
    model = _fit(LEGOLtngVelocity(_step_cond_config(one_step_euler_unlock=0.5)), train=False)
    with _at_fraction(0.0):
        predicted, _ = _steps_asked(model)
    assert predicted == {1.0, 0.5, 0.25, 0.125}


def test_unlocked_curriculum_is_the_plain_ladder() -> None:
    """Once every rung is unlocked the loss equals the run without a curriculum, draw for draw (same seed)."""
    from legofmt.cfm.flow_map import one_step_euler_loss

    plain = _fit(LEGOLtngVelocity(_step_cond_config()))
    cur = _fit(LEGOLtngVelocity(_step_cond_config(one_step_euler_unlock=0.5)))
    cur.load_state_dict(plain.state_dict())
    ds = _fake_batch(B=64)
    base, pid = plain.gen_base_wrapper(ds), plain.convert_pdgids(ds.f.pdgids)
    losses = []
    for model, f in ((plain, 0.0), (cur, 0.75), (cur, 0.0)):
        torch.manual_seed(0)
        with _at_fraction(f):
            losses.append(one_step_euler_loss(model, base, ds, pid))
    assert torch.equal(losses[0], losses[1])
    assert not torch.equal(losses[0], losses[2])   # the flag acts


def test_curriculum_logs_the_top_rung() -> None:
    """A curriculum run logs ``flow_map/top_rung``, the largest step's rung it draws (the proof the flag is active);
    a run without one logs nothing new."""
    for u, expected in ((0.5, 3.0), (0.0, None)):
        model = _fit(LEGOLtngVelocity(_step_cond_config(one_step_euler_unlock=u)))
        seen = {}
        with _at_fraction(0.0), patch.object(model, "_log_step", seen.update):
            model._step(_fake_batch(), 0)
        top = seen.get("flow_map/top_rung")
        assert (top is None) if expected is None else (float(top) == expected), f"u={u}: logged {top}"


def test_one_step_euler_requires_step_cond() -> None:
    with pytest.raises(ValueError, match="step_cond"):
        LEGOLtngVelocity(_tiny_config(one_step_euler_fac=1.0))


def test_trained_checkpoint_without_step_cond_loads() -> None:
    """The ``step_cond`` check is a requirement of training only: the ``rp_fm_v*`` checkpoints carry a non-zero
    ``one_step_euler_fac`` without ``step_cond`` and must still load for inference."""
    cfg = _tiny_config(one_step_euler_fac=1.0)
    cfg["state_dict"] = LEGOLtngVelocity(_tiny_config()).model.vf.state_dict()
    LEGOLtngVelocity(cfg)


def test_one_step_euler_every_gates_the_term() -> None:
    """With ``one_step_euler_every`` 2, an odd ``global_step`` skips the flow-map term and its three field calls."""
    model = _fit(LEGOLtngVelocity(_step_cond_config(one_step_euler_every=2)))
    seen = []
    orig = model.model.vf.forward
    model.model.vf.forward = lambda *a, _o=orig, **k: (seen.append(1), _o(*a, **k))[1]

    torch.manual_seed(0); model._step(_fake_batch(), 0)
    on = len(seen)                      # global_step 0: the velocity loss, two teacher half steps, the prediction
    seen.clear()
    with patch.object(LEGOLtngVelocity, "global_step", 1):
        torch.manual_seed(0); model._step(_fake_batch(), 0)
    off = len(seen)                     # global_step 1: the velocity loss only
    assert on == 4 and off == 1, f"expected 4 forwards on, 1 off; got {on}, {off}"


@pytest.mark.parametrize("learned_loss_weights", [False, True], ids=["step_gain", "lv_flow"])
@pytest.mark.parametrize("gs", [0, 1])
def test_every_param_gets_grad_even_when_flow_map_is_gated_off(gs, learned_loss_weights) -> None:
    """``step_gain`` is in the graph only when the field is given a ``d``. With ``one_step_euler_every`` > 1 the
    flow-map term is skipped on some steps, so the velocity loss must pass ``d = 0`` itself: otherwise ``step_gain``
    has no gradient on those steps and plain ``ddp`` (``find_unused_parameters=False``) raises. The same holds for
    ``lv_flow``: its ``fac * lv_flow`` term must be in the loss on every step."""
    cfg = _step_cond_config(one_step_euler_every=2, **({"learned_loss_weights": True} if learned_loss_weights else {}))
    model = _fit(LEGOLtngVelocity(cfg))
    with patch.object(LEGOLtngVelocity, "global_step", gs):
        model._step(_fake_batch(), 0).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"global_step={gs} left params without grad: {missing}"


def test_cond_init_scale_and_zero_biases() -> None:
    """The embedding is the mean of three maps, indexed by the token's mask, type and pdgid, so each weight needs std
    sqrt(3) x Xavier for the mean to be a Xavier map from ``in_dim`` to ``h_dim`` (``xavier_normal_`` on the 4-D
    tensors takes its fan from dims 0 and 1 and comes out 17x too low). The biases start at zero: Xavier-sized random
    biases are ~4x the signal from ``x``."""
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


def test_flow_map_under_learned_loss_weights() -> None:
    """Unweighted, the flow-map loss is ~1/2000 of the velocity loss, so it is normalised by a log-variance of its
    own (``lv_flow``) rather than by a hand-set factor."""
    model = LEGOLtngVelocity(_step_cond_config(learned_loss_weights=True))
    assert hasattr(model, "lv_flow") and model.lv_flow.ndim == 1
    ids = {id(p) for gr in model.opt.param_groups for p in gr["params"]}
    assert id(model.lv_flow) in ids, "lv_flow never reached the optimizer"
    _fit(model)._step(_fake_batch(), 0).backward()
    assert model.lv_flow.grad is not None and float(model.lv_flow.grad.abs()) > 0

    # exp(lv_flow) must track E[L_fm]: drive it with a constant flow-map loss
    const = 0.05
    with patch("legofmt.main.train_step.one_step_euler_loss", lambda *a, **k: torch.tensor(const)):
        opt = torch.optim.Adam([model.lv_flow], lr=0.1)
        for _ in range(400):
            opt.zero_grad()
            model._step(_fake_batch(), 0).backward()
            opt.step()
    got = float(model.lv_flow.detach().exp())
    assert abs(got - const) / const < 0.2, f"exp(lv_flow)={got:.4f}, expected ~{const}"


def _overflow_batch(B: int = 64) -> DataStruct:
    """Returns a batch in which the first half of the events has every outgoing energy at exactly 0."""
    ds = _fake_batch(B=B)
    ds.f.full[: B // 2, 3:, 0] = 0.0
    ds.f.full[B // 2:, 3:, 0] = 0.5
    return ds


def test_overflow_delta_zero_is_a_noop() -> None:
    plain, off = _twins(_tiny_config(overflow_delta=0.0), train=True)
    ds = _overflow_batch()
    torch.manual_seed(0); a = plain._step(ds, 0)
    torch.manual_seed(0); b = off._step(ds, 0)
    assert torch.equal(a, b), f"{a.item()} != {b.item()}"


def test_overflow_target_shift_is_value_scoped() -> None:
    model = LEGOLtngVelocity(_tiny_config(overflow_delta=0.08))
    ds = _overflow_batch()
    out = model._shift_overflow_targets(ds)
    e_new, e_old = _F(out.f.full).out_p[..., 0], _F(ds.f.full).out_p[..., 0]
    on_peak = e_old == 0
    assert torch.all(e_new[on_peak] == -0.08), "peak targets not shifted"
    assert torch.all(e_new[~on_peak] == e_old[~on_peak]), "off-peak targets moved"
    edep_new, edep_old = _F(out.f.full).edep, _F(ds.f.full).edep
    assert torch.all(edep_new[edep_old == 0] == -0.08), "zero edep not shifted"
    assert torch.all(edep_new[edep_old != 0] == edep_old[edep_old != 0]), "non-zero edep moved"
    npf = model.rc.n_prefix
    cond_tokens = [0, *range(2, npf + 1)]
    assert torch.equal(out.f.full[:, cond_tokens], ds.f.full[:, cond_tokens]), "prefix/incoming touched"
    assert ds.f.full[0, 3, 0] == 0.0, "input batch was mutated in place"


def test_overflow_leaves_the_base_untouched() -> None:
    """``overflow_delta`` must not change the base: ``gen_base_wrapper`` returns the same ``x_0`` with the shift on
    and off."""
    plain, ov = _twins(_tiny_config(overflow_delta=0.08), train=False)
    ds = _overflow_batch(B=128)
    torch.manual_seed(0); b_plain = plain.gen_base_wrapper(ds)
    torch.manual_seed(0); b_ov = ov.gen_base_wrapper(ov._shift_overflow_targets(ds))
    assert torch.equal(b_plain, b_ov), "the base moved"
    assert torch.all(_F(b_ov).out_p[..., 0] > 0), "negative base energies appeared"


def test_overflow_negative_energy_decodes_to_the_peak() -> None:
    """``to_mev`` clamps: a generated energy below 0 decodes exactly like an energy of 0."""
    from legofmt.geometry.energy_proj import EnergyProjections

    pen = EnergyProjections(cutoff_mev=10.0, max_energy=300.0)
    e_in = torch.tensor([[0.9]])
    at_zero = pen.to_mev(torch.tensor([[0.0]]), e_in)
    for over in (-0.01, -0.08, -0.5):
        assert torch.equal(pen.to_mev(torch.tensor([[over]]), e_in), at_zero)


def test_step_cond_euler_step_budgets_differ() -> None:
    model = _fit(LEGOLtngVelocity(_step_cond_config()), train=False)
    ds = _fake_batch(B=4)
    base = model.gen_base_wrapper(ds)
    one = model.solve(ds, x_init=base, method="euler", step_size=1.0)
    four = model.solve(ds, x_init=base, method="euler", step_size=0.25)
    assert one.shape == four.shape == base.shape
    assert torch.isfinite(one).all() and torch.isfinite(four).all()
    assert not torch.allclose(one, four), "step budget had no effect"


def test_step_cond_loads_legacy_state_dict() -> None:
    """A checkpoint without ``step_gain`` leaves it at 0, where the field ignores ``d``: the checkpoint keeps its
    velocity field instead of gaining a broken flow map. ``freqs_d`` is a non-persistent buffer: it is in no
    ``state_dict``, and one saved by an old checkpoint is not loaded."""
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

    # an old step_cond checkpoint holds freqs_d as a persistent buffer: an unexpected key, not loaded
    stale = {**sd, "freqs_d": torch.zeros_like(bank)}
    res = model.model.vf.load_state_dict(stale, strict=False)
    assert res.unexpected_keys == ["freqs_d"], res.unexpected_keys
    assert torch.equal(model.model.vf.freqs_d, bank), "stale bank overwrote the new one"


def test_uncert_zero_init_matches_unweighted() -> None:
    """At ``lv = 0`` every weight is exp(0) = 1 and the ``+ lv`` term is 0, so ``learned_loss_weights`` cannot change
    where training starts. Not bit-exact: the weighted branch multiplies before it reduces, which reorders the
    float32 sum."""
    plain, unc = _twins(_tiny_config(learned_loss_weights=True), train=True)
    ds = _fake_batch()
    torch.manual_seed(0); l_plain = plain._step(ds, 0)
    torch.manual_seed(0); l_unc = unc._step(ds, 0)
    assert torch.allclose(l_plain, l_unc, rtol=1e-6, atol=0), \
        f"{l_plain.item()} != {l_unc.item()}"


def test_uncert_param_is_1d_so_muon_skips_it() -> None:
    """``muon_factory`` routes ``ndim >= 2`` to Muon, whose Newton-Schulz step discards the magnitude: fatal for a
    log-variance, so ``lv`` stays 1-D."""
    from legofmt.optimizers import muon_factory

    model = LEGOLtngVelocity(_tiny_config(learned_loss_weights=True))
    assert model.lv.ndim == 1 and model.lv.numel() == 3
    # the tiny config uses schedulefree, so build a Muon the way opt: muon would
    opt = muon_factory([*model.model.parameters(), model.lv], lr=1e-3)
    muon = [gr for gr in opt.param_groups if gr.get("use_muon")]
    adamw = [gr for gr in opt.param_groups if not gr.get("use_muon")]
    assert muon and adamw, "expected both a Muon and an AdamW group"
    assert not any(p is model.lv for gr in muon for p in gr["params"]), "lv went to Muon"
    assert any(p is model.lv for gr in adamw for p in gr["params"]), "lv reached no group"


def test_uncert_lv_receives_grad_and_tracks_loss_scale() -> None:
    """The ``+ lv`` term drives ``exp(lv)`` to E[L], so a loss cell with a larger loss ends at a larger variance."""
    model = _fit(LEGOLtngVelocity(_tiny_config(learned_loss_weights=True)))
    ds = _fake_batch(B=8)
    opt = torch.optim.Adam([model.lv], lr=0.2)
    scales = None
    for _ in range(150):
        opt.zero_grad()
        sq = torch.zeros(8, 5, 7)
        sq[..., 0:1] = 4.0      # energy cell:    large loss
        sq[..., 1:4] = 1.0      # direction cell: medium
        sq[..., 4:7] = 0.25     # position cell:  small
        loss, _ = model.reduce_loss(sq, ds)
        loss.backward()
        opt.step()
        scales = model.lv.detach().exp()
    assert model.lv.grad is not None and model.lv.grad.abs().sum() > 0
    assert scales[0] > scales[1] > scales[2], f"variances not ordered: {scales}"
    # exp(lv) lands on each cell's mean squared error
    assert torch.allclose(scales, torch.tensor([4.0, 1.0, 0.25]), rtol=0.1), scales


def test_uncert_floor_caps_the_weight() -> None:
    """A near-zero loss would send the weight ``exp(-lv)`` to infinity; ``max_loss_weight`` is the floor of ``lv``
    that bounds it. The floor is applied to the parameter in place, so a cell below it keeps its gradient instead of
    staying stuck there (a ``clamp`` in the graph has zero gradient below the floor)."""
    model = _fit(LEGOLtngVelocity(_tiny_config(learned_loss_weights=True, max_loss_weight=-2.0)))
    with torch.no_grad():
        model.lv.fill_(-20.0)
    loss, _ = model.reduce_loss(torch.full((8, 5, 7), 1e-8), _fake_batch(B=8))
    loss.backward()
    assert torch.equal(model.lv.detach(), torch.full_like(model.lv, -2.0))
    assert (model.lv.grad != 0).all(), "a cell on the floor must keep its gradient"


def test_uncert_runs_on_the_time_independent_path() -> None:
    """The loss weights are per loss cell, not per ``t`` bin, so ``LEGOLtngDirect``, which has no ``t``, trains with
    them."""
    model = _fit(LEGOLtng(_tiny_config(learned_loss_weights=True)))     # LEGOLtng here is LEGOLtngDirect
    model._step(_fake_batch(), 0).backward()
    assert model.lv.grad is not None and model.lv.grad.abs().sum() > 0


def test_max_loss_weight_flow_is_its_own_floor() -> None:
    """A learned loss weight settles at ``lv = log(L)``, and the flow-map loss runs three decades below the velocity
    cells, so ``max_loss_weight_flow`` gives ``lv_flow`` a floor of its own."""
    m = LEGOLtngVelocity(_tiny_config({"step_cond": True}, learned_loss_weights=True, one_step_euler_fac=1.0,
                                      max_loss_weight_flow=-10.0))
    assert m.rc.max_loss_weight == -6.0 and m.rc.max_loss_weight_flow == -10.0
    _fit(m)
    with torch.no_grad():
        m.lv_flow.fill_(-20.0)
    m._step(_fake_batch(), 0).backward()  # global_step 0: the flow-map term fires
    assert float(m.lv_flow) == -10.0 and float(m.lv_flow.grad) != 0.0
    plain = LEGOLtngVelocity(_tiny_config(learned_loss_weights=True))
    assert plain.rc.max_loss_weight_flow == plain.rc.max_loss_weight, "default must follow max_loss_weight"


def test_legacy_loss_weight_keys_migrate() -> None:
    """A legacy checkpoint carries ``uncert_*`` keys and an ``lv`` table of (t bins, 3 cells). The keys are renamed,
    and the table is reduced to the three loss cells through the mean of its variances, because a converged cell
    sits at ``lv = log(L)``."""
    from legofmt.config import migrate_loss_weight_keys

    mc = {"uncert_weighting": True, "uncert_min": -4.0, "uncert_min_flow": -9.0,
          "uncert_bins": 2,
          "uncert_lv": {"lv": torch.tensor([1.0, 2.0, 4.0, 3.0, 2.0, 8.0]).log()}}
    out = migrate_loss_weight_keys(mc)
    assert out["learned_loss_weights"] is True
    assert out["max_loss_weight"] == -4.0 and out["max_loss_weight_flow"] == -9.0
    assert not {"uncert_bins", "uncert_lv", "uncert_weighting"} & set(out)
    assert torch.allclose(out["loss_weights"]["lv"].exp(), torch.tensor([2.0, 2.0, 6.0]))


def test_legacy_config_resolves_and_builds() -> None:
    """End to end: the legacy key ``uncert_weighting`` alone still switches the learned loss weights on, with a
    3-cell ``lv``."""
    model = LEGOLtngVelocity(_tiny_config(uncert_weighting=True, uncert_bins=16))
    assert model.rc.learned_loss_weights and model.lv.numel() == 3
