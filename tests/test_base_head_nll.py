"""The E_dep base fit must recover (mu, sigma) of a known logit-normal.

The previous ``l_edep`` matched the base's first two moments to per-event
targets; the width signal entered only through the fourth-order-small
``(m2 - t^2)^2`` term, which made sigma collapse (base std 7x too narrow in
``rp_fm_kin_020926``) and the whole fit unstable -- measured in
``repos/neurips_2026/base_head_isolated``. The fix is the logit-normal NLL,
the proper scoring rule for exactly the density the sampler draws from
(``E_dep = e_dep_max * sigmoid(mu + sig*z)``) and the same formula
``lego_eval/likelihood.py`` scores with. Zero-deposit events are excluded:
the logit-normal has no atom, and the zero-deposit atom is the flow's job
(``edep_overflow_delta``).
"""

from __future__ import annotations

import torch

from legofmt.data.struct import DataStruct, _F
from legofmt.main.modules import LEGOLtng

MU_TRUE, SIG_TRUE = -2.0, 0.5


def _config() -> dict:
    pdgids = torch.tensor([11, 22, 2212], dtype=torch.int64).sort().values
    return {
        "state_dict": {},
        "config": {
            "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
            "val_conf": {"val_frac": 0.01, "seed": 0},
            "base_conf": {
                "base_range": 3.4, "kappa": torch.tensor(8.0), "bs_frac": 0.0,
                "base_dist": "poles", "scale_dist": "sm_norm", "tanh_theta": True,
            },
            "model_conf": {
                "manifold": [
                    {"name": "euclidean", "dim": 1},
                    {"name": "sphere", "dim": 3},
                    {"name": "sphere", "dim": 3},
                ],
                "max_energy": 300.0,
                "pdgids": pdgids,
                "cond_scalars": ("Density", "Z", "A", "Size"),
                # > 0 so __init__ builds base_head; the test drives
                # pretrain_base itself, so the count is otherwise unused.
                "base_pretrain_batches": 1,
                "model_args": {
                    "h_dim": 16, "nlayers": 1, "nhead": 2, "in_dim": 7,
                    "max_seq_l": 8, "ntypes": 4, "nvtypes": 2,
                    "npdgids": pdgids.numel() + 1, "ff_mult": 1, "dropout": 0.0,
                    "use_adaptive_rmsnorm": True, "use_adaptive_layerscale": True,
                    "ff_swish": True, "ff_glu": True,
                },
            },
            "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
        },
    }


def _batch(B: int, edep: torch.Tensor) -> DataStruct:
    """Fixed conditioning: rows [Density, edep, Z, A, Size, incoming, 2 x out]."""
    f = torch.zeros(B, 8, 8)
    for row, val in ((0, 3.0), (2, 18.0), (3, 39.95), (4, 100.0)):
        f[:, row, 0] = val
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 1, 0] = edep
    f[:, 5, 0] = 1.0
    f[:, 5, 1:4] = torch.tensor([1.0, 0.0, 0.0])
    f[:, 5, 4:7] = torch.tensor([-1.0, 0.0, 0.0])
    f[:, 5, 7] = 22.0
    g = torch.Generator().manual_seed(0)
    f[:, 6:, 0] = torch.rand(B, 2, generator=g)
    f[:, 6:, 1:4] = torch.nn.functional.normalize(torch.randn(B, 2, 3, generator=g), dim=-1)
    f[:, 6:, 4:7] = torch.nn.functional.normalize(torch.randn(B, 2, 3, generator=g), dim=-1)
    f[:, 6:, 7] = 11.0
    m = torch.zeros(B, 8, dtype=torch.long)
    m[:, 1] = 1
    m[:, 6:] = 1
    am = torch.ones(B, 8, dtype=torch.bool)
    return DataStruct(f, m, am)


def _batches(n: int, bs: int = 512):
    g = torch.Generator().manual_seed(1)
    for _ in range(n):
        edep = torch.sigmoid(MU_TRUE + SIG_TRUE * torch.randn(bs, generator=g))
        yield _batch(bs, edep)


def test_pretrain_recovers_logit_normal_params() -> None:
    model = LEGOLtng(_config())
    model.pretrain_base(_batches(300))
    _, mu, sig, _ = model.base_head_params(_batch(4, torch.full((4,), 0.1)))
    mu, sig = float(mu.mean()), float(sig.mean())
    assert abs(mu - MU_TRUE) < 0.3, f"mu {mu} vs {MU_TRUE}"
    assert abs(sig - SIG_TRUE) / SIG_TRUE < 0.3, f"sig {sig} vs {SIG_TRUE}"


def test_zero_deposit_events_do_not_break_the_loss() -> None:
    model = LEGOLtng(_config())
    model.model.train()
    model.gen_base_wrapper(_batch(64, torch.zeros(64)))
    loss = model._base_dist_loss
    assert loss is not None and torch.isfinite(loss), loss
    loss.backward()
    for p in model.base_head.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
