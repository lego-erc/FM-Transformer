"""The moment head's E_dep fit recovers ``(mu, sig)`` of a known logit-normal (legacy ``_base_moment_loss``).

E_dep's prior is fitted by the logit-normal NLL, the density the sampler draws from
(``E_dep = e_dep_max * sigmoid(mu + sig*z)``; ``lego_eval/likelihood.py`` scores with the same formula). When two
moments are matched instead, the width enters only through a fourth-order term: ``sig`` collapses (base std 7x too
narrow in ``rp_fm_kin_020926``) and the fit is unstable. Events without a deposit are excluded: the logit-normal has
no point mass at 0, and that point mass is the flow's (``overflow_delta``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test_helpers import _head_batch, _head_config
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng

MU_TRUE, SIG_TRUE = -2.0, 0.5


def _config() -> dict:
    cfg = _head_config(False)
    del cfg["config"]["base_conf"]["base_head_nout"]
    return cfg


def _batch(edep: torch.Tensor) -> DataStruct:
    """Returns a batch with fixed conditioning: an incoming photon and two outgoing particles in every event."""
    return _head_batch(edep, torch.full(edep.shape, 2), pid_in=22.0)


def _batches(n: int, bs: int = 512):
    g = torch.Generator().manual_seed(1)
    for _ in range(n):
        edep = torch.sigmoid(MU_TRUE + SIG_TRUE * torch.randn(bs, generator=g))
        yield _batch(edep)


def test_pretrain_recovers_logit_normal_params() -> None:
    model = LEGOLtng(_config())
    model.pretrain_base(_batches(300))
    _, mu, sig, _, _, _ = model.base_head_params(_batch(torch.full((4,), 0.1)))
    mu, sig = float(mu.mean()), float(sig.mean())
    assert abs(mu - MU_TRUE) < 0.3, f"mu {mu} vs {MU_TRUE}"
    assert abs(sig - SIG_TRUE) / SIG_TRUE < 0.3, f"sig {sig} vs {SIG_TRUE}"


def test_pretraining_runs_every_requested_batch_past_one_epoch() -> None:
    # base_pretrain_batches counts batches, across epochs of the loader. Capped at one epoch, 1000 asked were 280 run
    # on the 4.6M-event wsi64 set at bs 16384: a head fitted on a single pass.
    cfg = _config()
    cfg["config"]["model_conf"].update(base_pretrain_batches=5, base_pretrain_bs=4)
    cfg["config"]["dl_conf"]["num_workers"] = 0
    model = LEGOLtng(cfg)
    model._train_ds = _batch(torch.full((8,), 0.1))   # two batches per epoch
    model._trainer = SimpleNamespace(is_global_zero=True, strategy=SimpleNamespace(
        root_device=torch.device("cpu"), broadcast=lambda obj, src=0: obj))
    seen = []
    model.pretrain_base = lambda batches, **kw: seen.extend(batches) or 0.0
    model._pretrain_base_if_needed()
    assert len(seen) == 5


def test_pretraining_anneals_the_lr_over_the_requested_steps(monkeypatch) -> None:
    # ``steps`` cosine-anneals the lr from 1e-2 to 1e-4. On the wsi64 E^-0.1 set a 128x2 head annealed over 4000 steps
    # fitted the 1 GeV proton survivors to mu +0.016, sig 0.09 (shipped: +0.19, 0.27); held-out NLL -2.01 vs -0.79.
    made = []

    class Adam(torch.optim.Adam):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
    monkeypatch.setattr(torch.optim, "Adam", Adam)
    LEGOLtng(_config()).pretrain_base(_batches(10), steps=10)
    assert made[0].param_groups[0]["lr"] == pytest.approx(1e-4)


def test_zero_deposit_events_do_not_break_the_loss() -> None:
    model = LEGOLtng(_config())
    model.model.train()
    model.gen_base_wrapper(_batch(torch.zeros(64)))
    loss = model._base_head_loss
    assert loss is not None and torch.isfinite(loss), loss
    loss.backward()
    for p in model.base_head.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
