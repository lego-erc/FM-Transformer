"""legofmt.rng.EventRNG: opt-in per-event counter-based random numbers for generation. Every draw is a pure function
of (seed, event key, draw site, element), so an event's randomness does not depend on its batch, its row, or which
other events a caller subsets -- and two models generating the same keyed events share it (common random numbers)."""

from __future__ import annotations

import copy

import pytest
import torch

from legofmt.base_dist.gen_base import GenerateBase
from legofmt.cfm.edep_ae import EdepLayerAE
from legofmt.data.struct import DataStruct, set_layout
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from legofmt.rng import EventRNG, species_rank_ids
from test_edep_latents import _batch as _latent_batch, _model as _latent_model, _wake
from test_helpers import _flow_config, _gun_cond, _pt_mult_config, _save_mult_ckpt


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    """A velocity flow and a pass-through-head count model, both drawing a cube transform (canon_sym + sym_aug)."""
    tmp = tmp_path_factory.mktemp("rng_ckpts")
    torch.manual_seed(0)
    cfg = _flow_config()
    cfg["model_conf"].update(canon_sym=True, sym_aug=True)
    m = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": m.model.vf.state_dict(), "config": cfg}, tmp / "flow.pt")
    torch.manual_seed(0)
    mult = _save_mult_ckpt(tmp, _pt_mult_config(passthrough_head=True, canon_sym=True, sym_aug=True))
    return GenerateOut(str(tmp / "flow.pt"), str(mult), device="cpu")


@pytest.fixture(autouse=True)
def _layout(generator: GenerateOut) -> None:
    set_layout(generator.cond_names)


def _mixed_cond(n: int = 24) -> torch.Tensor:
    """Photons (the pass-through head covers them) and protons, interleaved."""
    return torch.stack((_gun_cond(22, batch=n), _gun_cond(2212, batch=n)), 1).reshape(2 * n, -1)


def test_draws_depend_on_the_key_not_the_row() -> None:
    keys = torch.tensor([3, 7, 11, 19, 23])
    rows = torch.tensor([4, 1, 3])
    full, sub = EventRNG(keys, seed=5), EventRNG(keys, seed=5).sub(rows)
    other = EventRNG(keys[rows], seed=5)
    probs = torch.tensor([[0.2, 0.0, 0.5, 0.3]]).expand(len(keys), 4)
    per_row = torch.arange(10).view(5, 2)
    for draw in (lambda r: r.uniform(1), lambda r: r.normal(2, torch.arange(6)),
                 lambda r: r.randint(8, 3), lambda r: r.categorical(probs[:len(r.keys)], 4, 2)):
        assert torch.equal(draw(full)[rows], draw(sub)) and torch.equal(draw(sub), draw(other))
    assert torch.equal(full.uniform(6, per_row)[rows], sub.uniform(6, per_row[rows]))
    assert not torch.equal(full.uniform(1), EventRNG(keys, seed=6).uniform(1))     # the seed matters
    assert not torch.equal(full.uniform(1), full.uniform(2))                       # so does the draw site


def test_draws_have_the_requested_distributions() -> None:
    r = EventRNG(torch.arange(200_000), seed=0)
    u, z = r.uniform(1, torch.arange(2)), r.normal(2)
    assert 0 < u.min() and u.max() < 1
    assert abs(u.mean() - 0.5) < 3e-3 and abs(u.var() - 1 / 12) < 2e-3
    assert abs(torch.corrcoef(u.T)[0, 1]) < 1e-2                                   # elements are independent
    assert abs(z.mean()) < 1e-2 and abs(z.var() - 1) < 1e-2
    assert abs(((z - z.mean()) ** 4).mean() / z.var() ** 2 - 3) < 0.05
    k = r.randint(8, 3)
    assert k.min() == 0 and k.max() == 7 and (torch.bincount(k, minlength=8) / len(k) - 1 / 8).abs().max() < 3e-3
    p = torch.tensor([0.2, 0.0, 0.5, 0.3])
    c = r.categorical(p.expand(200_000, 4), 4)
    assert (torch.bincount(c, minlength=4) / len(c) - p).abs().max() < 4e-3 and (c != 1).all()
    many = r.categorical(p.expand(200_000, 4), 5, torch.arange(3))                 # several draws per row
    assert many.shape == (200_000, 3)
    assert (torch.bincount(many.flatten(), minlength=4) / many.numel() - p).abs().max() < 4e-3


def test_duplicate_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="unique"):
        EventRNG(torch.tensor([1, 2, 2]))


def test_species_rank_ids_are_unique_and_survive_an_inserted_particle() -> None:
    pid = torch.tensor([[1, 1, 2, 2, 2, 0, 0], [2, 1, 2, 1, 0, 0, 0]])
    ids = species_rank_ids(pid)
    assert all(row.unique().numel() == row.numel() for row in ids)
    grown = torch.tensor([[1, 1, 1, 2, 2, 2, 0]])        # one more species-1 particle, ahead of the species-2 ones
    assert torch.equal(species_rank_ids(grown)[0, 3:6], ids[0, 2:5])


@pytest.mark.parametrize("scale_dist", ["trunc_norm", "uniform", "sm_norm", "logit_norm"])
def test_keyed_energy_prior_matches_the_torch_one(scale_dist: str) -> None:
    gb = GenerateBase({"base_conf": {"scale_dist": scale_dist}})
    B, e_in = 100_000, torch.ones(100_000, 1)
    keyed = gb.rd_scale((B, 2), e_in, rng=EventRNG(torch.arange(B)), elem=torch.arange(2).expand(B, 2))
    torch.manual_seed(0)
    plain = gb.rd_scale((B, 2), e_in)
    assert keyed.shape == plain.shape
    assert abs(keyed.mean() - plain.mean()) < 4e-3 and abs(keyed.std() - plain.std()) < 4e-3


@torch.no_grad()
def test_keyed_generation_does_not_depend_on_batch_layout(generator: GenerateOut) -> None:
    cond, keys = _mixed_cond(), torch.arange(48) + 1000
    sols, mask, _ = generator(cond, rng=EventRNG(keys, seed=3))
    assert 0 < (mask.sum(-1) == 0).sum() < len(cond)       # some pass-throughs, some interactions: both branches ran
    perm = torch.randperm(len(cond), generator=torch.Generator().manual_seed(0))
    parts = [generator(cond[p], rng=EventRNG(keys[p], seed=3)) for p in perm.tensor_split(3)]
    shuffled, smask = torch.empty_like(sols), torch.empty_like(mask)
    shuffled[perm], smask[perm] = torch.cat([s for s, _, _ in parts]), torch.cat([m for _, m, _ in parts])
    assert torch.equal(smask, mask)
    torch.testing.assert_close(shuffled, sols, equal_nan=True)
    again, _, _ = generator(cond, rng=EventRNG(keys, seed=4))
    assert not torch.equal(again.nan_to_num(), sols.nan_to_num())


@torch.no_grad()
def test_keyed_base_pairs_slots_by_species_and_rank(generator: GenerateOut) -> None:
    """An extra particle of one species leaves every other slot's x_0 as it was."""
    cm = generator._model_cond(_gun_cond(2212, batch=1), False)
    pt = torch.zeros(1, dtype=torch.bool)
    draws = {}
    for name, counts in (("base", torch.tensor([[1, 2]])), ("extra", torch.tensor([[2, 2]]))):
        ds = DataStruct(*generator.build_batch(cm, pt, counts))
        ds_t, _ = generator.model._prep_solve(ds)
        x0 = generator.model.gen_base_wrapper(ds_t, rng=EventRNG(torch.tensor([7])))
        draws[name] = (x0, ds_t.f.out_p[0, :, -1])
    (b, pid_b), (e, pid_e) = draws["base"], draws["extra"]
    n0 = generator.n_prefix + 1
    pions_b, pions_e = n0 + (pid_b == pid_b[1]).nonzero()[:, 0], n0 + (pid_e == pid_b[1]).nonzero()[:, 0]
    torch.testing.assert_close(e[0, pions_e], b[0, pions_b])
    torch.testing.assert_close(e[0, n0], b[0, n0])           # the first photon, too


@torch.no_grad()
def test_keyed_counts_couple_two_count_models(generator: GenerateOut) -> None:
    """Inverse-CDF draws from shared uniforms: a slightly different count model mostly draws the same counts."""
    mult_in, pidx = generator._mult_in(generator._model_cond(_mixed_cond(200), False))
    other = copy.deepcopy(generator.gen_mult)
    other.proj_out_b.data += 0.2 * torch.randn(other.proj_out_b.shape, generator=torch.Generator().manual_seed(0))
    keys = torch.arange(len(pidx))
    a = generator.gen_mult((mult_in, None, pidx), rng=EventRNG(keys))
    same = (a == other((mult_in, None, pidx), rng=EventRNG(keys))).all(-1).float().mean()
    unpaired = (a == other((mult_in, None, pidx), rng=EventRNG(keys, seed=1))).all(-1).float().mean()
    assert same < 1.0 and same > unpaired + 0.2


@torch.no_grad()
def test_keyed_edep_layers_depend_on_the_key_not_the_row() -> None:
    torch.manual_seed(0)
    ae = EdepLayerAE(6, 3, hidden=8, n_mix=2)
    z, tot = torch.randn(8, 3), torch.rand(8) * 100
    rng, rows = EventRNG(torch.arange(8) + 50), torch.tensor([6, 0, 3])
    whole = ae.sample(z, tot, rng=rng)
    torch.testing.assert_close(ae.sample(z[rows], tot[rows], rng=rng.sub(rows)), whole[rows])
    assert not torch.equal(ae.sample(z, tot, rng=EventRNG(torch.arange(8))), whole)


@torch.no_grad()
def test_keyed_latents_follow_their_events_through_length_buckets() -> None:
    m = _latent_model()
    _wake(m)
    ds = _latent_batch()
    ds = DataStruct(ds.f.full, ds.m.full, ds.am.full.clone())
    ds.am.full[0, -1] = False                       # two lengths: the bucketed solve reorders the rows
    base = m.gen_base_wrapper(ds)
    keys = torch.tensor([10, 20])
    m._solve_by_length(ds, base, rng=EventRNG(keys), step_size=0.5, method="euler")
    plain = m._edep_latents
    m.rc.odeint_conf["len_buckets"] = [4]
    m._solve_by_length(ds, base, rng=EventRNG(keys), step_size=0.5, method="euler")
    torch.testing.assert_close(m._edep_latents, plain)


def test_generate_refuses_keys_of_the_wrong_length(generator: GenerateOut) -> None:
    with pytest.raises(ValueError, match="one key per event"):
        generator(_mixed_cond(2), rng=EventRNG(torch.arange(3)))
