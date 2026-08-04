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


def test_step_cond_zero_is_the_velocity_slice() -> None:
    """d=0 must reproduce the unconditioned field exactly -- that is what makes
    an existing velocity checkpoint a valid warm start for the flow map."""
    model = LEGOLtngVelocity(_step_cond_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch()
    torch.manual_seed(0); out_none = _vf_call(model, ds, t=torch.tensor(0.3))
    torch.manual_seed(0); out_zero = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(0.0))
    model.model.vf.step_cond = False
    torch.manual_seed(0); out_off = _vf_call(model, ds, t=torch.tensor(0.3), d=torch.tensor(0.5))
    assert torch.equal(out_none, out_zero), "d=0 differs from d=None"
    assert torch.equal(out_none, out_off), "step_cond=False slice differs from d=0"


def test_step_cond_distinguishes_t_from_d() -> None:
    """Regression for 96344ee: emb_t(t) + emb_d(d) built on one frequency bank
    with the same sin/cos interleave is exactly symmetric under swapping t and
    d, so the model cannot tell (t=0.2, d=0.8) from (t=0.8, d=0.2)."""
    model = LEGOLtngVelocity(_step_cond_config())
    model.on_fit_start(); model.eval()
    ds = _fake_batch()
    a = _vf_call(model, ds, t=torch.tensor(0.2), d=torch.tensor(0.8))
    b = _vf_call(model, ds, t=torch.tensor(0.8), d=torch.tensor(0.2))
    assert not torch.allclose(a, b), "(t, d) is degenerate under swapping"


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
    """A pre-step_cond checkpoint has no freqs_d; strict=False must absorb that
    and leave the constructed bank intact."""
    plain = LEGOLtngVelocity(_tiny_config())
    sd = plain.model.vf.state_dict()
    assert "freqs_d" not in sd
    model = LEGOLtngVelocity(_step_cond_config())
    bank = model.model.vf.freqs_d.clone()
    res = model.model.vf.load_state_dict(sd, strict=False)
    assert res.missing_keys == ["freqs_d"], res.missing_keys
    assert not res.unexpected_keys, res.unexpected_keys
    assert torch.equal(model.model.vf.freqs_d, bank)


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
