"""base_conf.ang_species: one angular prior (kappa, and pow with ang_shape) per outgoing species.

The pooled prior lands between the species: photons came out over-dispersed while electrons sat
at the null. Per-slot moment matching lets each species get its own opening-angle distribution.
"""

from __future__ import annotations

import math

import pytest
import torch


def _rc(**bc):
    class _RC:
        pdgids_template = torch.tensor([11, 22])
        cond_scalars = ("Density", "Z", "A", "Size")
        base_pretrain_batches = 1
        config = {"base_conf": {"base_head": None, **bc}}
    return _RC()


class _GB:
    scale_dist = "sm_norm"
    sm_scale = 0.5
    kappa = 8.0
    tanh_theta = True
    e_dep_max = 1.0


@pytest.mark.parametrize("ang_shape", [False, True])
def test_head_width_one_angular_block_per_species(ang_shape) -> None:
    from legofmt.base_dist.base_nn import build_base_head

    n_cls = 3  # two template species + the unknown class
    head = build_base_head(_rc(ang_species=True, ang_shape=ang_shape), _GB())
    assert head[-1].out_features == 3 + n_cls * (1 + int(ang_shape))
    b = head[-1].bias.exp()
    assert torch.allclose(b[3:3 + n_cls], torch.full((n_cls,), 8.0))     # every species starts at the prior kappa
    if ang_shape:
        assert torch.allclose(b[3 + n_cls:], torch.ones(n_cls))         # and at pow = 1
    plain = build_base_head(_rc(ang_shape=ang_shape), _GB())
    assert plain[-1].out_features == 4 + int(ang_shape)                 # unchanged without the flag


def test_per_slot_prior_recovers_species_specific_kappas() -> None:
    """Two species with different opening angles in every event: fitting the per-slot branch of
    _base_moment_loss recovers one kappa per species, while the pooled branch can only settle on a
    compromise between them."""
    from legofmt.base_dist.base_nn import BaseDist
    from legofmt.data.struct import DataStruct

    class M(BaseDist):
        def __init__(self):
            self.rc = _rc(ang_species=True, ang_weight=1000.0)
            self.gen_base = _GB()

    torch.manual_seed(0)
    n, p = 256, 6
    full = torch.zeros(n, 2 + 1 + p, 8)
    full[..., 1, 0] = 0.3
    full[:, 2, 1:4] = torch.tensor([1.0, 0.0, 0.0])      # incoming direction +x
    full[:, 2, 4:7] = torch.tensor([1.0, 0.0, 0.0])
    # slots 0-2: narrow species (theta ~ 0.05 pi), slots 3-5: wide species (theta ~ 0.3 pi)
    theta = torch.cat((torch.full((n, 3), 0.05), torch.full((n, 3), 0.30)), 1) * math.pi
    phi = torch.rand(n, p) * 2 * math.pi
    d = torch.stack((theta.cos(), theta.sin() * phi.cos(), theta.sin() * phi.sin()), -1)
    full[:, 3:, 1:4] = d
    full[:, 3:, 4:7] = d
    ds = DataStruct(full, torch.ones(n, 3 + p, dtype=torch.long), torch.ones(n, 3 + p, dtype=torch.long))
    fwd = torch.ones(n, dtype=torch.bool)
    s, mu, sig = torch.full((n, 1), 0.5), torch.zeros(n, 1), torch.ones(n, 1)
    m = M()
    z = 2 ** 0.5 * torch.erfinv(torch.linspace(-0.995, 0.995, 400))
    mean_angle = lambda k: float((z.abs() / k).tanh().mean())
    grid = torch.linspace(1.0, 60.0, 600)
    k_n = float(grid[torch.tensor([abs(mean_angle(k) - 0.05) for k in grid]).argmin()])
    k_w = float(grid[torch.tensor([abs(mean_angle(k) - 0.30) for k in grid]).argmin()])

    def fit(per_slot):
        lk = torch.full((2 if per_slot else 1,), math.log(8.0), requires_grad=True)
        opt = torch.optim.Adam([lk], lr=0.05)
        for _ in range(400):
            k = lk.exp()
            kap = torch.cat((k[0].expand(n, 3), k[1].expand(n, 3)), 1) if per_slot else k[0].expand(n, 1)
            loss = m._base_moment_loss(ds, fwd, s, mu, sig, kap)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return lk.exp().detach()

    k_split = fit(True)
    assert abs(float(k_split[0]) - k_n) / k_n < 0.15, (float(k_split[0]), k_n)
    assert abs(float(k_split[1]) - k_w) / k_w < 0.15, (float(k_split[1]), k_w)
    k_pool = float(fit(False)[0])
    assert k_w < k_pool < k_n                                            # the compromise sits in between
