"""``base_conf.base_head_nout``: the outgoing set as base-head conditioning.

E_dep is bimodal in the outgoing multiplicity, and the head was blind to it.
Measured on ``rp_had_060926`` at the eval point (argon, rho ~ 10, size 100 mm,
proton T > 950 MeV, chord ~ 1.0, n = 262): the ``n_out == 1`` pass-through
branch is 46% of events with a relative width ``(q90-q10)/median`` of 0.121,
against 1.816 for the mixture. One logit-normal cannot be both, so the fitted
prior spanned the mixture and the flow -- which contracts the prior only ~5x --
inherited the excess width.

The multiplicity and the per-pdgid composition are fixed before the flow runs
(``gen_batch`` sizes the slots from the mult model), so this is conditioning
the sampler genuinely has, not a leak of the target.
"""

from __future__ import annotations

import torch

from legofmt.data.struct import DataStruct, _F
from legofmt.main.modules import LEGOLtng

MU1, SIG1 = -2.5, 0.15   # n_out == 1: the sharp pass-through spike
MU2, SIG2 = -0.5, 0.80   # n_out == 2: the broad inelastic continuum


def _config(nout: bool) -> dict:
    pdgids = torch.tensor([11, 22, 2212], dtype=torch.int64).sort().values
    return {
        "state_dict": {},
        "config": {
            "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
            "val_conf": {"val_frac": 0.01, "seed": 0},
            "base_conf": {
                "base_range": 3.4, "kappa": torch.tensor(8.0), "bs_frac": 0.0,
                "base_dist": "poles", "scale_dist": "sm_norm", "tanh_theta": True,
                "base_head_nout": nout,
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


def _batch(edep: torch.Tensor, n_out: torch.Tensor) -> DataStruct:
    """Rows [Density, edep, Z, A, Size, incoming, out, out]; ``n_out`` in {1, 2}."""
    B = len(edep)
    f = torch.zeros(B, 8, 8)
    for row, val in ((0, 3.0), (2, 18.0), (3, 39.95), (4, 100.0)):
        f[:, row, 0] = val
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 1, 0] = edep
    f[:, 5, 0] = 1.0
    f[:, 5, 1:4] = torch.tensor([1.0, 0.0, 0.0])
    f[:, 5, 4:7] = torch.tensor([-1.0, 0.0, 0.0])
    f[:, 5, 7] = 2212.0
    g = torch.Generator().manual_seed(0)
    f[:, 6:, 0] = torch.rand(B, 2, generator=g)
    f[:, 6:, 1:4] = torch.nn.functional.normalize(torch.randn(B, 2, 3, generator=g), dim=-1)
    f[:, 6:, 4:7] = torch.nn.functional.normalize(torch.randn(B, 2, 3, generator=g), dim=-1)
    f[:, 6:, 7] = 11.0
    second = n_out == 2
    f[:, 7, 7] = torch.where(second, torch.tensor(11.0), torch.tensor(0.0))
    m = torch.zeros(B, 8, dtype=torch.long)
    m[:, 1] = 1
    m[:, 6:] = 1
    am = torch.ones(B, 8, dtype=torch.bool)
    am[:, 7] = second
    return DataStruct(f, m, am)


def _mixed(bs: int, seed: int) -> DataStruct:
    g = torch.Generator().manual_seed(seed)
    n_out = torch.where(torch.rand(bs, generator=g) < 0.5,
                        torch.tensor(1), torch.tensor(2))
    mu = torch.where(n_out == 1, MU1, MU2)
    sig = torch.where(n_out == 1, SIG1, SIG2)
    return _batch(torch.sigmoid(mu + sig * torch.randn(bs, generator=g)), n_out)


def _n_features(model) -> int:
    return model.base_head[0].weight.shape[1]


def test_in_features_grow_only_with_the_flag() -> None:
    off, on = LEGOLtng(_config(False)), LEGOLtng(_config(True))
    n_pdg = len(off.pdgids_template)
    assert _n_features(off) == 4 + n_pdg
    # + log1p(n_out) + log1p(count) per class, classes = pdgids + pad
    assert _n_features(on) == 4 + n_pdg + n_pdg + 2


def test_out_set_feats_counts_multiplicity_and_species() -> None:
    model = LEGOLtng(_config(True))
    n_out = torch.tensor([1, 2, 1, 2])
    n, cnt = model._out_set_feats(_batch(torch.full((4,), 0.1), n_out))
    assert torch.allclose(n.squeeze(-1), n_out.float().log1p())
    # every valid outgoing slot is pdgid 11, so one column carries the whole count
    assert torch.allclose(cnt.max(-1).values, n_out.float().log1p())
    assert int((cnt > 0).sum(-1).max()) == 1, "only the pdgid-11 column is filled"


def test_a_head_saved_without_the_flag_is_not_mis_loaded() -> None:
    saved = LEGOLtng(_config(False)).base_head.state_dict()
    cfg = _config(True)
    cfg["config"]["base_conf"]["base_head"] = saved
    model = LEGOLtng(cfg)
    n_pdg = len(model.pdgids_template)
    assert _n_features(model) == 4 + 2 * n_pdg + 2
    assert not torch.allclose(model.base_head[0].weight[:, : 4 + n_pdg],
                              saved["0.weight"]), "stale input layer was reused"


def test_nout_conditioning_separates_the_bimodal_edep() -> None:
    """With the flag the head resolves the two modes; without it, it cannot."""
    got = {}
    for flag in (False, True):
        model = LEGOLtng(_config(flag))
        model.pretrain_base(_mixed(512, s) for s in range(300))
        probe = torch.tensor([1, 2])
        _, mu, sig, _ = model.base_head_params(_batch(torch.full((2,), 0.1), probe))
        got[flag] = (mu.flatten().tolist(), sig.flatten().tolist())

    (mu1_off, mu2_off), _ = got[False]
    (mu1_on, mu2_on), (sig1_on, sig2_on) = got[True]

    assert abs(mu1_off - mu2_off) < 0.05, (
        f"blind head somehow split the modes: {mu1_off} vs {mu2_off}")
    assert mu1_on < mu2_on - 1.0, f"mu not separated: {mu1_on} vs {mu2_on}"
    assert abs(mu1_on - MU1) < 0.4 and abs(mu2_on - MU2) < 0.4, (mu1_on, mu2_on)
    assert sig1_on < sig2_on / 2, f"sigma not separated: {sig1_on} vs {sig2_on}"
    assert abs(sig1_on - SIG1) / SIG1 < 0.6, f"spike width {sig1_on} vs {SIG1}"
