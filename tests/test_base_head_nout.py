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

from test_helpers import _head_batch as _batch, _head_config as _config
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng

MU1, SIG1 = -2.5, 0.15   # n_out == 1: the sharp pass-through spike
MU2, SIG2 = -0.5, 0.80   # n_out == 2: the broad inelastic continuum


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
        _, mu, sig, _, _ = model.base_head_params(_batch(torch.full((2,), 0.1), probe))
        got[flag] = (mu.flatten().tolist(), sig.flatten().tolist())

    (mu1_off, mu2_off), _ = got[False]
    (mu1_on, mu2_on), (sig1_on, sig2_on) = got[True]

    assert abs(mu1_off - mu2_off) < 0.05, (
        f"blind head somehow split the modes: {mu1_off} vs {mu2_off}")
    assert mu1_on < mu2_on - 1.0, f"mu not separated: {mu1_on} vs {mu2_on}"
    assert abs(mu1_on - MU1) < 0.4 and abs(mu2_on - MU2) < 0.4, (mu1_on, mu2_on)
    assert sig1_on < sig2_on / 2, f"sigma not separated: {sig1_on} vs {sig2_on}"
    assert abs(sig1_on - SIG1) / SIG1 < 0.6, f"spike width {sig1_on} vs {SIG1}"


def test_only_edep_sees_the_outgoing_set() -> None:
    """sm_scale and kappa must not move with n_out: conditioning them rewrote the
    outgoing priors 8x across branches and cost 2x on the direction losses."""
    model = LEGOLtng(_config(True))
    model.pretrain_base(_mixed(512, s) for s in range(150))
    s, mu, sig, kap, _ = model.base_head_params(_batch(torch.full((2,), 0.1), torch.tensor([1, 2])))
    assert abs(float(mu[0] - mu[1])) > 0.5, "E_dep prior should separate the branches"
    assert torch.allclose(s[0], s[1]) and torch.allclose(kap[0], kap[1]), (
        "sm_scale / kappa leaked n_out")
