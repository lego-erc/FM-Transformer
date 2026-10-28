"""``model_conf.edep_whiten``: the flow transports the E_dep row in units of the event's own base prior.

The converged field (midpoint/rk4) never resolves an E_dep narrower than ~0.0022 log-row units (IQR) -- a fixed
floor in the channel, measured on the K = 1 W/Si stack over 600-1000 MeV protons at 0-20 deg, where Geant4's
survivor peak is 0.0012 wide; where the true width is larger (40 deg) the field is right. A survivor sits at
u ~ 0.9 with a 0.1% spread, so the floor is 1.6-2x its width. Whitening with the calibrated head,
``y = (logit(u / e_dep_max) - mu) / sig``, turns the prior into N(0, 1) and that peak into ~N(-0.4, 0.4^2):
the same structure at O(1) in the network's coordinate. Deposit-free targets (the overflow sentinel) map to a
fixed -6, decoded back to 0; ``solve`` whitens on entry and undoes it on exit, so every caller sees u.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from test_helpers import _head_batch, _head_config, _tiny_config
from legofmt.main.modules import LEGOLtng


def _model() -> LEGOLtng:
    torch.manual_seed(0)    # the rest of the head is drawn from the global RNG: some draws put 0.3 out of reach
    cfg = _head_config(True)
    cfg["config"]["model_conf"]["edep_whiten"] = True
    m = LEGOLtng(cfg)
    with torch.no_grad():   # a per-event prior: the zero-init head would give every event mu = 0, sig = 1
        m.base_head[-1].weight.copy_(0.5 * torch.randn(m.base_head[-1].weight.shape,
                                                       generator=torch.Generator().manual_seed(3)))
    return m


def test_whitening_round_trips_deposits_and_keeps_zeros_zero() -> None:
    model = _model()
    edep = torch.tensor([0.3, 0.7, 0.95, 0.0])
    ds = _head_batch(edep, torch.tensor([1, 2, 1, 2]))
    y = model._edep_whiten(ds.f.full, ds)
    assert y[3, 1, 0] < -4.5
    back = model._edep_whiten(y, ds, inverse=True)
    assert torch.allclose(back[:3, 1, 0], edep[:3], atol=1e-5) and back[3, 1, 0] == 0
    back[:, 1, 0] = ds.f.full[:, 1, 0]
    assert torch.equal(back, ds.f.full)  # only the E_dep cell moves


def test_a_deposit_far_below_its_prior_stays_a_deposit() -> None:
    # 0.13% of real deposits sit more than 4.5 prior sigmas below mu: decoded as 0 they would turn into pass-throughs
    # (E_dep == 0) for the count model, so the continuum is held at 4.4 sigmas
    model = _model()
    ds = _head_batch(torch.full((2,), 0.5), torch.ones(2, dtype=torch.long))
    _, mu, sig, _, _ = model.base_head_params(ds)
    edep = model.gen_base.e_dep_max * torch.sigmoid(mu.reshape(-1) + torch.tensor([-5.0, -8.0]) * sig.reshape(-1))
    ds = _head_batch(edep, torch.ones(2, dtype=torch.long))
    back = model._edep_whiten(model._edep_whiten(ds.f.full, ds), ds, inverse=True)[:, 1, 0]
    assert (back > 0).all(), back


def test_the_base_draw_whitens_to_its_latent() -> None:
    model = _model()
    ds = _head_batch(torch.full((4096,), 0.5), torch.ones(4096, dtype=torch.long))
    torch.manual_seed(0)
    y = model._edep_whiten(model.gen_base_wrapper(ds), ds)[:, 1, 0]
    assert abs(float(y.mean())) < 0.1 and abs(float(y.std()) - 1) < 0.1


def test_solve_hands_back_the_deposit_scale() -> None:
    # a field pushing the E_dep cell by +1: whitened, that is one prior sigma and u stays below e_dep_max;
    # on the u scale it would leave the channel
    model = _model()
    ds = _head_batch(torch.full((8,), 0.5), torch.ones(8, dtype=torch.long))

    def push(x, t, *a, **k):
        v = torch.zeros_like(x)
        v[:, 1, 0] = 1.0
        return v
    model.model.forward = push
    out = model.solve(ds, x_init=model.gen_base_wrapper(ds), method="euler", step_size=0.5)
    e = out[:, 1, 0]
    assert ((e > 0) & (e < model.gen_base.e_dep_max)).all(), e


def test_capped_scale_keeps_broad_priors_on_a_finer_grid() -> None:
    # edep_whiten_max_sig caps the divisor, narrow priors unchanged. Tried against the K = 4 (27 mm block) +35%
    # overshoot of 1 GeV p inelastic E_dep: it did not remove it (the unwhitened control overshoots too) and
    # re-broadened the K = 1 survivor peak
    model = _model()
    ds = _head_batch(torch.tensor([0.3, 0.7]), torch.tensor([1, 2]))
    full = model._edep_whiten(ds.f.full, ds)[:, 1, 0]
    _, mu, sig, _, _ = model.base_head_params(ds)
    model.rc = replace(model.rc, edep_whiten_max_sig=0.1)
    capped = model._edep_whiten(ds.f.full, ds)
    assert torch.allclose(capped[:, 1, 0], full * sig.reshape(-1) / sig.reshape(-1).clamp(max=0.1), atol=1e-5)
    assert torch.allclose(model._edep_whiten(capped, ds, inverse=True)[:, 1, 0], ds.f.full[:, 1, 0], atol=1e-5)


def test_feed_hands_the_prior_to_the_network_and_restores_the_filler() -> None:
    # a whitened row is only decodable with its (mu, sig); edep_whiten_feed puts them in the row's two direction
    # factors as unit vectors (mu, 1, 0) / (log sig, 1, 0) -- on the sphere, so the projection keeps them -- and the
    # inverse puts the 1s filler back
    model = _model()
    model.rc = replace(model.rc, edep_whiten_feed=True)
    ds = _head_batch(torch.tensor([0.3, 0.7]), torch.tensor([1, 2]))
    y = model._edep_whiten(ds.f.full, ds)
    _, mu, sig, _, _ = model.base_head_params(ds)
    enc = torch.nn.functional.normalize(torch.stack((mu.reshape(-1), torch.ones(2), torch.zeros(2)), -1), dim=-1)
    assert torch.allclose(y[:, 1, 1:4], enc, atol=1e-6)
    assert torch.allclose(y[:, 1, 4:7].norm(dim=-1), torch.ones(2), atol=1e-6)
    back = model._edep_whiten(y, ds, inverse=True)
    assert torch.allclose(back[:, 1, :7], ds.f.full[:, 1, :7], atol=1e-5)


def test_global_whitening_is_one_map_for_every_event() -> None:
    # edep_whiten_global [c, s] is the same logit map for every event, y = (logit u - c) / s (another K = 4 arm, see
    # above); below c - 4.4 s in logit (u < 0.33 for (1.5, 0.5), i.e. E_dep under ~1 keV on the 1e-6 log row) a
    # deposit is held there
    model = _model()
    model.rc = replace(model.rc, edep_whiten_global=(1.5, 0.5))
    edep = torch.tensor([0.4, 0.7, 0.9, 0.0])
    ds = _head_batch(edep, torch.tensor([1, 2, 1, 2]))
    y = model._edep_whiten(ds.f.full, ds)[:, 1, 0]
    assert torch.allclose(y[:3], (torch.logit(edep[:3]) - 1.5) / 0.5, atol=1e-5) and y[3] < -4.5
    back = model._edep_whiten(model._edep_whiten(ds.f.full, ds), ds, inverse=True)[:, 1, 0]
    assert torch.allclose(back, edep, atol=1e-5)


def test_likelihood_refuses_the_whitened_row() -> None:
    # the whitening's Jacobian is not in the ODE likelihood: a silent number would be wrong
    model = _model()
    ds = _head_batch(torch.full((4,), 0.5), torch.ones(4, dtype=torch.long))
    with pytest.raises(NotImplementedError, match="edep_whiten"):
        model.log_likelihood(ds, log_p0=lambda x: x.sum())


def test_whitening_needs_the_base_head() -> None:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["edep_whiten"] = True
    with pytest.raises(ValueError, match="edep_whiten"):
        LEGOLtng(cfg)
