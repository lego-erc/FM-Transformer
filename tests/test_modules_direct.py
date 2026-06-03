"""Smoke test for :class:`legofmt.main.modules.LEGOLtngDirect`.

Builds a tiny model from an in-memory config (no HF download, no
``meta.json``), constructs a synthetic batch matching the padded sequence
layout, and runs one forward + backward pass.
"""

from __future__ import annotations

import warnings

import torch

from legofmt.data.struct import DataStruct, _F
from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity, LEGOLtngDirect as LEGOLtng


def _has_nonzero_grad(params) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


def _tiny_config() -> dict:
    pdgids = torch.tensor([22, 211, 2212], dtype=torch.int64).sort().values
    return {
        "state_dict": {},  # take the checkpoint-restore branch (skips meta.json lookup)
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
                    {"name": "euclidean", "dim": 3},
                    {"name": "sphere",    "dim": 3},
                ],
                "pdgids": pdgids,
                "model_args": {
                    "h_dim": 16,
                    "nlayers": 2,
                    "nhead": 2,
                    "in_dim": 6,
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
    f[:, 0, 1] = 1.0                                # density
    f[:, 1, 1] = 0.1                                # edep
    _F(f).non_p[..., 2:-1] = 1.0                    # sphere-projx-safe placeholder
    f[:, 2, 1:4] = torch.tensor([0.0, 0.0, 150.0])  # incoming mom (E-scaled)
    f[:, 2, 4:7] = torch.tensor([0.0, 0.0, -1.0])   # incoming pos on sphere
    f[:, 2, 7] = 22.0                               # incoming pdgid
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 1:4] = torch.randn(B, L - 3, 3, generator=g) * 30.0
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
    assert out.shape == ds.f.full.shape[:2] + (6,)
    norms = out[:, 3:, 3:6].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms


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
    for split in (12, 6, 5):  # 5 -> partial last chunk (12 % 5 = 2)
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

    # Not in self.modules() -> Lightning's train/eval propagation skips it.
    assert not any(mod is teacher for mod in model.modules())

    # Not in self.parameters() -> DDP doesn't sync, optimizer never updates it.
    teacher_first_param = next(teacher.parameters())
    assert id(teacher_first_param) not in {id(p) for p in model.parameters()}

    # parent.train() must not flip teacher's training flag.
    assert teacher.model.training is False
    model.train()
    assert teacher.model.training is False
