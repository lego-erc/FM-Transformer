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
    multiplicity_metrics,
    survival_by_primary,
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
        n_prefix = 2

    pdgids_template = torch.tensor([-11, 11, 22])

    def gen_base_wrapper(self, ds_t):
        return ds_t.f.model_in.clone()

    def sample_passthrough(self, ds_t):
        return torch.zeros(len(ds_t), dtype=torch.bool)  # the gate never fires

    def solve(self, ds_t, x_init, return_species=False, **_kw):
        if not return_species:
            return x_init
        # identity on the kinematics, and the classes the fixtures below build
        # their pdgid column from ([0, 0, 0, 11, -11, 22, 11, 0] as 1-based
        # indices into pdgids_template), so one slot comes out empty
        sp = torch.tensor([0, 0, 0, 2, 1, 3, 2, 0]).expand(x_init.shape[0], -1)
        return x_init, sp


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
        | {"val/w1_mult", "val/mult_ratio", "val/zero_out_err", "val/pt_rate_err"}
        | {f"val/w1_particle/{n}" for n in KIN_NAMES}
        | {f"val/w1_summary/{n}" for n in SUMMARY_FEATURE_NAMES}
    )


def test_val_generation_applies_the_passthrough_gate():
    """Validation must score what GenerateOut emits: a fired gate means the primary leaves
    unchanged with no deposit. Sending those events through the flow instead (which is
    trained with weight 0 on them) inflated the neutron E_dep W1 9x on the stack run."""
    from legofmt.data.struct import _F
    from legofmt.main.modules import LEGOLtng
    from test_modules_direct import _tiny_config
    from test_passthrough_gate import _pt_batch

    torch.manual_seed(0)
    model = LEGOLtng(_tiny_config()).eval()
    with torch.no_grad():
        model.pt_head[-1].bias.fill_(50.0)
        model.pt_head[-1].weight.zero_()
    ds = _pt_batch(passthrough=True)                       # photon primaries: the gate may fire
    gen, active, pdg = ShowerValMetrics._generate(model, ds)
    assert active.sum(-1).eq(1).all() and active[:, 0].all()
    assert pdg[:, 0].eq(22).all()
    assert _F(gen).edep.abs().max().item() == 0.0
    first = _F(gen).out_p[:, 0]
    assert first[:, 0].abs().max().item() == 0.0            # e_out == e_in
    assert torch.allclose(first[:, 1:7], ds.f.in_cc[:, 0, 1:7])


def test_edep_ratio_multi_selects_each_side_on_its_own_multiplicity():
    """A unified model generates the composition, so the fake event at an index is not the
    real one's: selecting the fake side on the REAL n_out mixes absorbed fake events into the
    multi cell. Here the multi events agree exactly and only the absorbed ones differ."""
    edep_real = torch.full((64,), 0.4)
    edep_fake = torch.cat([torch.full((32,), 0.4), torch.full((32,), 0.8)])
    pdgid_in = torch.full((64,), 211)
    n_real = torch.full((64,), 4)
    n_fake = torch.cat([torch.full((32,), 4), torch.zeros(32, dtype=torch.long)])
    out = edep_by_primary(edep_real, edep_fake, pdgid_in, n_real, n_fake)
    assert abs(out["val/edep_ratio_multi/pip"].item() - 1.0) < 1e-6


def test_zero_out_and_passthrough_rate_are_separate_metrics():
    """`passthrough_err` used to measure P(n_out == 0), which is not a pass-through."""
    active = torch.zeros(4, 3, dtype=torch.bool)
    active[0, 0] = active[1, 0] = active[2, :2] = True     # real: two singles, one pair, one empty
    g_active = active.clone()
    g_active[3, 0] = True                                  # fake: the empty one gets a particle
    pdg = torch.full((4, 3), 22.0)
    edep = torch.tensor([0.0, 0.3, 0.3, 0.3])               # real: event 0 is a pass-through
    g_edep = torch.tensor([0.0, 0.0, 0.3, 0.0])             # fake: events 0, 1, 3 look like one
    out = multiplicity_metrics(active, pdg, g_active, pdg, edep, g_edep)
    assert abs(out["val/zero_out_err"].item() - (0.0 - 0.25)) < 1e-6
    assert abs(out["val/pt_rate_err"].item() - (0.75 - 0.25)) < 1e-6


def test_survival_by_primary_catches_a_dropped_leading_particle():
    """The stack run lost the leading pi+ in ~1 of 5 events while every pooled metric moved
    by a few percent: the primary's own species, counted per gun, is where it shows."""
    pdgid_in = torch.tensor([211] * 4 + [11] * 4)
    pdg = torch.tensor([[211.0, 22.0]] * 4 + [[11.0, 22.0]] * 4)
    active = torch.ones(8, 2, dtype=torch.bool)
    g_active = active.clone()
    g_active[:2] = False                                   # two pi+ events lose everything
    out = survival_by_primary(pdgid_in, active, pdg, g_active, pdg)
    assert abs(out["val/primary_count_ratio/pip"].item() - 0.5) < 1e-6
    assert abs(out["val/zero_out_err/pip"].item() - 0.5) < 1e-6
    assert out["val/primary_count_ratio/em"].item() == 1.0
    assert all(k.startswith(UNSYNCED_PREFIXES) for k in out)
