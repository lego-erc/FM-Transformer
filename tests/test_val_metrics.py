import torch

from legofmt.data.struct import DataStruct
from legofmt.mod_comps.val_metrics import (
    KIN_NAMES,
    SUMMARY_FEATURE_NAMES,
    ShowerValMetrics,
    compute_mmd,
    event_summary,
    particle_kinematics,
    standardize,
    w1_per_feature,
)


def test_mmd_identical_is_zero():
    X = torch.randn(256, 5)
    assert compute_mmd(X, X.clone()).item() < 1e-5


def test_mmd_separated_is_positive():
    X = torch.randn(256, 5)
    Y = torch.randn(256, 5) + 5.0
    assert compute_mmd(X, Y).item() > 0.1


def test_w1_identical_is_zero():
    X = torch.randn(128, 7)
    w1 = w1_per_feature(X, X.clone())
    assert w1.shape == (7,)
    assert torch.allclose(w1, torch.zeros(7), atol=1e-6)


def test_w1_shift_matches_mean_gap():
    X = torch.randn(512, 1)
    assert torch.allclose(w1_per_feature(X, X + 3.0), torch.tensor([3.0]), atol=1e-5)


def test_standardize_zero_mean_unit_std():
    a = torch.randn(100, 4) * 5 + 2
    b = torch.randn(100, 4) * 5 + 2
    both = torch.cat(standardize(a, b))
    assert torch.allclose(both.mean(0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(both.std(0), torch.ones(4), atol=1e-2)


def test_summary_shape_matches_names():
    B, K = 8, 12
    pdg = torch.tensor([11, -11, 22, 0]).repeat(B, K // 4 + 1)[:, :K]
    active = torch.ones(B, K, dtype=torch.bool)
    s = event_summary(torch.randn(B, K, 3), torch.randn(B, K, 3), pdg, active, torch.rand(B))
    assert s.shape == (B, len(SUMMARY_FEATURE_NAMES)) == (B, 31)


def test_particle_kinematics_packs_active():
    B, K = 4, 6
    active = torch.zeros(B, K, dtype=torch.bool)
    active[:, :3] = True
    feats = particle_kinematics(torch.randn(B, K, 3), torch.randn(B, K, 3), active)
    assert feats.shape == (B * 3, len(KIN_NAMES))


def test_summary_identical_metrics_zero():
    B, K = 64, 10
    pdg = torch.full((B, K), 11)
    active = torch.ones(B, K, dtype=torch.bool)
    s = event_summary(torch.randn(B, K, 3), torch.randn(B, K, 3), pdg, active, torch.rand(B))
    sa, sb = standardize(s, s.clone())
    assert compute_mmd(sa, sb).item() < 1e-5
    assert torch.allclose(w1_per_feature(sa, sb), torch.zeros(31), atol=1e-6)


class _StubLego:
    """Minimal LEGOLtng surface used by ShowerValMetrics (solve = identity flow)."""

    device = "cpu"

    class rc:
        odeint_conf = {"step_size": 0.5}

    def gen_base_wrapper(self, ds_t):
        return ds_t.f.model_in.clone()

    def solve(self, ds_t, x_init, **_kw):
        return x_init


def test_shower_val_metrics_returns_loggable_dict():
    B, L = 16, 8
    f = torch.randn(B, L, 8)
    f[..., 7] = torch.tensor([0, 0, 0, 11, -11, 22, 11, 0.0])
    am = torch.zeros(B, L, dtype=torch.bool)
    am[:, 2:] = True
    mm = am.clone().long()
    mm[:, :3] = 0
    ds = DataStruct(f, mm, am)

    out = ShowerValMetrics()(_StubLego(), ds)
    assert {"val/mmd_particle", "val/mmd_summary"} <= out.keys()
    assert sum(k.startswith("val/w1_particle/") for k in out) == len(KIN_NAMES)
    assert sum(k.startswith("val/w1_summary/") for k in out) == len(SUMMARY_FEATURE_NAMES)
    assert all(torch.isfinite(v) for v in out.values())
