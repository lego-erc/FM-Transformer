"""Reflow targets keep the prefix rows' direction filler, on both training paths.

The E_dep row is generated (``mask == 1``) but its direction columns are the
``(1, 1, 1)`` filler; letting the teacher transport them gave that row an
off-manifold target (the ``loss/one_step_euler ~ 430`` spikes of kin_opts_290826).
"""

from __future__ import annotations

import torch

from legofmt.distill.reflow import LEGOLtngDirect, _teacher_from_state
from legofmt.main.modules import LEGOLtng
from test_modules_direct import _fake_batch, _tiny_config


def _with_teacher(cls, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda m, **k: m)  # inductor would cost ~10 s
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["reflow_kwargs"] = {"method": "midpoint", "step_size": 0.5}
    m = cls(cfg)
    teacher = _teacher_from_state(cfg["config"], LEGOLtng(_tiny_config()).model.vf.state_dict())
    object.__setattr__(m, "reflow_teacher", teacher)
    ds = _fake_batch(B=4)
    ds.m.full[:, 1] = 1  # production: the E_dep row is generated
    return m, ds


def test_velocity_reflow_target_keeps_prefix_filler(monkeypatch) -> None:
    m, ds = _with_teacher(LEGOLtng, monkeypatch)
    seen, sample = {}, m.ps.sample

    def spy(base, data, t):
        seen["data"] = data
        return sample(base, data, t)

    m.ps.sample = spy
    m.train()
    m._step(ds, 0)
    tgt = seen["data"]
    assert torch.equal(tgt[:, :m.rc.n_prefix, 1:], ds.f.non_cc[..., 1:])
    assert not torch.equal(tgt[:, m.rc.n_prefix + 1:], ds.f.out_cc), "teacher target not used"


def test_direct_reflow_target_keeps_prefix_filler(monkeypatch) -> None:
    m, ds = _with_teacher(LEGOLtngDirect, monkeypatch)
    tgt = m._reflow_target(ds, m.gen_base_wrapper(ds))
    assert torch.equal(tgt[:, :m.rc.n_prefix, 1:], ds.f.non_cc[..., 1:])
