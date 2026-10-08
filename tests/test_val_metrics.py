import torch

from legofmt.data.struct import DataStruct
from legofmt.log_metrics.val_metrics import (
    KIN_NAMES,
    SUMMARY_FEATURE_NAMES,
    ShowerValMetrics,
    UNSYNCED_PREFIXES,
    compute_mmd,
    edep_by_primary,
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
    s = event_summary(
        torch.randn(B, K, 3), torch.rand(B, K, 1), torch.randn(B, K, 3),
        pdg, active, torch.rand(B),
    )
    assert s.shape == (B, len(SUMMARY_FEATURE_NAMES)) == (B, 31)


def test_particle_kinematics_packs_active():
    B, K = 4, 6
    active = torch.zeros(B, K, dtype=torch.bool)
    active[:, :3] = True
    feats = particle_kinematics(
        torch.randn(B, K, 3), torch.rand(B, K, 1), torch.randn(B, K, 3), active,
    )
    assert feats.shape == (B * 3, len(KIN_NAMES))


def test_summary_identical_metrics_zero():
    B, K = 64, 10
    pdg = torch.full((B, K), 11)
    active = torch.ones(B, K, dtype=torch.bool)
    s = event_summary(
        torch.randn(B, K, 3), torch.rand(B, K, 1), torch.randn(B, K, 3),
        pdg, active, torch.rand(B),
    )
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


def test_edep_by_primary_splits_the_guns():
    """A proton-only E_dep error must show up on the proton keys and nowhere else."""
    edep_real = torch.cat([torch.full((64,), 0.4), torch.full((64,), 0.4)])
    edep_fake = edep_real.clone()
    edep_fake[64:] *= 1.25                       # protons only, +25%
    pdgid_in = torch.cat([torch.full((64,), 11), torch.full((64,), 2212)])
    n_out = torch.full((128,), 4)

    out = edep_by_primary(edep_real, edep_fake, pdgid_in, n_out)
    assert set(out) == {"val/w1_edep/em", "val/w1_edep/p",
                        "val/edep_ratio_multi/em", "val/edep_ratio_multi/p"}
    assert out["val/w1_edep/em"].item() == 0.0
    assert out["val/edep_ratio_multi/em"].item() == 1.0
    assert abs(out["val/edep_ratio_multi/p"].item() - 1.25) < 1e-5


def test_edep_by_primary_skips_thin_and_single_particle_cells():
    edep = torch.rand(40)
    pdgid_in = torch.full((40,), 2112)
    # 8 events with a secondary is under the 32-event floor -> no ratio key
    n_out = torch.cat([torch.full((32,), 1), torch.full((8,), 3)])
    out = edep_by_primary(edep, edep.clone(), pdgid_in, n_out)
    assert set(out) == {"val/w1_edep/n"}


def test_population_dependent_keys_are_excluded_from_sync():
    """Every key whose presence depends on the shard must match UNSYNCED_PREFIXES.

    validation_step logs with sync_dist for everything else, and a synced key that
    one rank emits and another does not leaves the ranks issuing different numbers
    of collectives -- NCCL then hangs at the sanity check and times out 30 min later.
    """
    both = edep_by_primary(                         # e- and p, both with a multi cell
        torch.rand(128), torch.rand(128),
        torch.cat([torch.full((64,), 11), torch.full((64,), 2212)]),
        torch.full((128,), 4),
    )
    thin = edep_by_primary(                         # protons only, all n_out == 1
        torch.rand(64), torch.rand(64),
        torch.full((64,), 2212), torch.ones(64),
    )
    varying = set(both) ^ set(thin)
    assert varying, "fixture no longer exercises a varying key set"
    assert all(k.startswith(UNSYNCED_PREFIXES) for k in varying), sorted(varying)


def test_static_val_metric_keys_are_still_synced():
    B, L = 16, 8
    f = torch.randn(B, L, 8)
    f[..., 7] = torch.tensor([0, 0, 0, 11, -11, 22, 11, 0.0])
    am = torch.zeros(B, L, dtype=torch.bool)
    am[:, 2:] = True
    mm = am.clone().long()
    mm[:, :3] = 0

    out = ShowerValMetrics()(_StubLego(), DataStruct(f, mm, am))
    synced = {k for k in out if not k.startswith(UNSYNCED_PREFIXES)}
    assert synced == (
        {"val/mmd_particle", "val/mmd_summary"}
        | {f"val/w1_particle/{n}" for n in KIN_NAMES}
        | {f"val/w1_summary/{n}" for n in SUMMARY_FEATURE_NAMES}
    )
