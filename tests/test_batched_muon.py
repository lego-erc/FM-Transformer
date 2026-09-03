"""BatchedMuon must reproduce pytorch_optimizer.Muon's update (Muon on >=2-D incl.
flattened 4-D weights, AdamW on 1-D, decoupled weight decay) up to bf16
Newton-Schulz rounding, on the parameter shapes the vector field actually has."""

import copy

import pytest
import torch
from pytorch_optimizer import Muon

from legofmt.mod_comps.optimizers import BatchedMuon, muon_factory

KW = dict(lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=7e-3, weight_decouple=True,
          adamw_lr=3e-3, adamw_betas=(0.9, 0.999), adamw_wd=1e-2, adamw_eps=1e-8)


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 32), (32, 64), (64, 32), (16, 2, 24, 7), (2, 2, 24, 7), (1, 64), (64,), (1,), (24,)]
    return [torch.nn.Parameter(torch.randn(*s, generator=g)) for s in shapes]


def _groups(ps):
    return [{"params": [p for p in ps if p.ndim >= 2], "use_muon": True},
            {"params": [p for p in ps if p.ndim < 2], "use_muon": False}]


def _run(opt, ps, steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g) * 0.1
        opt.step()


@pytest.mark.parametrize("decouple", [True, False])
def test_matches_pytorch_optimizer_muon(decouple):
    kw = {**KW, "weight_decouple": decouple}
    a, b = _params(), _params()
    ref, new = Muon(_groups(a), **kw), BatchedMuon(_groups(b), **kw)
    a0 = [p.detach().clone() for p in a]
    _run(ref, a, 3); _run(new, b, 3)
    for p_ref, p_new, p0 in zip(a, b, a0):
        upd = (p_ref - p0).norm()
        assert upd > 0
        rel = (p_ref - p_new).norm() / upd
        tol = 2e-2 if p_ref.ndim >= 2 else 1e-5   # bf16 Newton-Schulz (bmm vs mm) vs fp32 AdamW
        assert rel < tol, (tuple(p_ref.shape), float(rel))


def test_factory_groups_and_scheduler_scales_both_lrs():
    ps = _params()
    opt = muon_factory(ps, **KW)
    assert [g["use_muon"] for g in opt.param_groups] == [True, False]
    assert opt.param_groups[0]["lr"] == KW["lr"] and opt.param_groups[1]["lr"] == KW["adamw_lr"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(KW["lr"] * 0.5)
    assert opt.param_groups[1]["lr"] == pytest.approx(KW["adamw_lr"] * 0.5)
