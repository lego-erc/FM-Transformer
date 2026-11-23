"""Tests for the length buckets: a batch runs at its own sequence length instead of ``max_seq_l``.

The field has no positional embedding and masks padding out of the attention and the loss, so cutting the padding off
a batch must not change the output of any real token.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DistributedSampler

from legofmt.data.load.loader import LengthBucketBatchSampler, merge_sparse
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng
from legofmt.config import resolve_legoltng_config, resolve_len_buckets
from legofmt.multiplicity.model import MultModel

from test_helpers import _fake_batch, _mult_config, _tiny_config

L_MAX = 8          # n_prefix 2 (Density, E_dep) + incoming + 5 outgoing slots
N_OUT = (1, 2, 5, 3, 1, 2)                   # event lengths 4, 5, 8, 6, 4, 5


def _config(dl=None, odeint=None, step_cond: bool = False, **model_conf) -> dict:
    cfg = _tiny_config({"max_seq_l": L_MAX, "ntypes": L_MAX, "step_cond": step_cond}, **model_conf)
    cfg["config"]["dl_conf"].update(bs=len(N_OUT), **(dl or {}))
    if odeint:
        cfg["config"]["odeint_conf"] = odeint
    return cfg


def _muon_cosine() -> dict:
    return {"opt": "muon", "lr": 1e-3, "scheduler": {"cls": "warmup_cosine", "total_steps": 1000, "warmup_frac": 0.1}}


def _ragged_batch(n_out=N_OUT, edep_generated: bool = False, L: int = L_MAX) -> DataStruct:
    """Returns ``_fake_batch`` at length ``L`` in which event i keeps only its first ``n_out[i]`` outgoing particles.

    The outgoing tokens after them are padding (``m = 0``, ``am = False``). Padding holds finite on-manifold filler
    and pdgid 0, like ``DataPrep``'s ``nan_to_num(1)`` tokens: zeros would project to NaN, and NaN * 0 survives a mask.
    """
    ds = _fake_batch(B=len(n_out), L=L, edep_generated=edep_generated)
    f, m, am = ds.f.full.clone(), ds.m.full.clone(), ds.am.full.clone()
    for i, n in enumerate(n_out):
        f[i, 3 + n:] = f[i, 3]
        f[i, 3 + n:, 7] = 0.0
        m[i, 3 + n:] = 0
        am[i, 3 + n:] = False
    return DataStruct(f, m, am)


def _model(cfg=None) -> LEGOLtng:
    torch.manual_seed(0)           # identical init for every config: they share one architecture
    model = LEGOLtng(cfg or _config())
    model.on_fit_start()
    model.eval()
    return model


def test_velocity_is_unchanged_by_trimming_the_padding() -> None:
    """What the length buckets rest on: the field's velocity on every real token is the same whether the batch runs
    at ``L_MAX`` or cut to its longest event."""
    model = _model()
    ds = _ragged_batch()
    x = model.gen_base_wrapper(ds)
    t = torch.rand(ds.f.full.shape[0], L_MAX)
    pdgids = model.convert_pdgids(ds.f.pdgids)
    full = model.model(x, t, ds.m.full, ds.am.full, model.types_embd, pdgids)

    short = torch.tensor([n <= 2 for n in N_OUT])     # longest kept event: 3 + 2 = 5 tokens
    L = 5
    trimmed = model.model(
        x[short, :L], t[short, :L], ds.m.full[short, :L], ds.am.full[short, :L],
        model.types_embd[:, :L], pdgids[short, :L],
    )
    real = ds.am.full[short, :L].unsqueeze(-1)
    torch.testing.assert_close(trimmed * real, full[short, :L] * real, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- configuration

def test_auto_buckets_are_four_per_octave_up_to_max_seq_l() -> None:
    assert resolve_len_buckets("auto", 134) == (8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 134)
    assert resolve_len_buckets("auto", 28) == (8, 10, 12, 14, 16, 20, 24, 28)
    assert resolve_len_buckets("auto", 8) == (8,)


def test_explicit_buckets_are_sorted_capped_and_end_at_max_seq_l() -> None:
    assert resolve_len_buckets([32, 8], 50) == (8, 32, 50)
    assert resolve_len_buckets([8, 64], 40) == (8, 40)
    assert resolve_len_buckets(None, 134) is None


def test_config_without_len_keys_resolves_to_off() -> None:
    rc = resolve_legoltng_config(_config())
    assert rc.len_buckets is None


def test_config_resolves_len_keys() -> None:
    rc = resolve_legoltng_config(_config(dl={"len_buckets": "auto", "len_split": 2},
                                         odeint={"len_buckets": [4, 6]}))
    assert rc.len_buckets == (8,) and rc.len_mode == "split" and rc.len_split == 2


@pytest.mark.parametrize("dl, match", [({"len_mode": "pack"}, "len_mode"), ({"len_split": 4}, "len_split")],
                         ids=["unknown_len_mode", "len_split_must_divide_bs"])
def test_bad_len_keys_raise(dl, match) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_legoltng_config(_config(dl={"len_buckets": "auto", **dl}))


# ---------------------------------------------------------------- split pieces

def _by_length(ds: DataStruct) -> DataStruct:
    """Returns the batch in length order.

    A split step sorts its batch by length before it samples anything, so the unsplit reference must see the events
    in the same order.
    """
    return ds[ds.am.full.sum(1).sort(stable=True).indices]


@pytest.mark.parametrize("n_out, width, pieces", [
    (N_OUT, 8, [(slice(0, 3), 6), (slice(3, 6), 8)]),       # lengths 4,5,8,6,4,5 -> 4,4,5 | 5,6,8
    ((1, 2, 1, 2), 6, [(slice(0, 2), 4), (slice(2, 4), 6)]),  # longest 5: the batch is trimmed to bucket 6
    ((1, 2, 1), 6, [(slice(0, 3), 6)]),                       # a final batch len_split does not divide: whole, unsorted
], ids=["sorted_equal_pieces", "trimmed_to_longest_bucket", "short_final_batch_whole"])
def test_split_sorts_and_cuts_equal_pieces_at_their_buckets(n_out, width, pieces) -> None:
    model = _model(_config(dl={"len_buckets": [4, 6], "len_split": 2}))
    ds, got = model._len_pieces(_ragged_batch(n_out))
    lengths = ds.am.full.sum(1).tolist()
    assert got == pieces and ds.am.full.shape[1] == width
    assert len(pieces) == 1 or lengths == sorted(lengths)     # a split batch is cut in length order


# ---------------------------------------------------------------- split training step

def _loss_and_grads(model: LEGOLtng, ds: DataStruct) -> tuple[torch.Tensor, dict]:
    model.train()
    model.zero_grad()
    torch.manual_seed(1)
    loss = model._step(ds, 0)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("flow_map", [False, True], ids=["velocity", "with_flow_map"])
def test_split_step_matches_the_unsplit_step(flow_map: bool) -> None:
    extra = dict(learned_loss_weights=True, edep_cell=True)
    if flow_map:
        extra.update(one_step_euler_fac=0.5, one_step_euler_sections=4)
    plain = _model(_config(step_cond=flow_map, **extra))
    split = _model(_config(dl={"len_buckets": [4, 6], "len_split": 2}, step_cond=flow_map, **extra))
    ds = _by_length(_ragged_batch(edep_generated=True))   # lengths 4,4,5 | 5,6,8 -> pieces at 6 and 8
    seen = {id(plain): set(), id(split): set()}
    for m in (plain, split):
        m.model.register_forward_pre_hook(lambda mod, args, key=id(m): seen[key].add(args[0].shape[1]))

    loss_p, grad_p = _loss_and_grads(plain, ds)
    loss_s, grad_s = _loss_and_grads(split, ds)
    assert seen[id(plain)] == {L_MAX} and seen[id(split)] == {6, 8}
    torch.testing.assert_close(loss_s, loss_p, rtol=1e-5, atol=1e-6)
    assert grad_s.keys() == grad_p.keys()
    for n in grad_p:
        torch.testing.assert_close(grad_s[n], grad_p[n], rtol=1e-4, atol=1e-6, msg=n)


# ---------------------------------------------------------------- group sampler

def test_group_sampler_batches_come_from_one_bucket_each_event_at_most_once() -> None:
    lengths = torch.randint(4, 9, (103,), generator=torch.Generator().manual_seed(0))
    buckets = (4, 6, 8)
    sampler = LengthBucketBatchSampler(lengths, buckets, bs=4, seed=0)
    batches = list(sampler)                    # lists of event indices, as torch's BatchSampler yields them
    assert len(batches) == len(sampler)
    seen = torch.tensor([i for b in batches for i in b])
    assert seen.unique().numel() == seen.numel()
    bucket_of = torch.bucketize(lengths, torch.tensor(buckets))
    for batch in batches:
        assert len(batch) == 4 and bucket_of[batch].unique().numel() == 1
    per_bucket = torch.bincount(bucket_of, minlength=3)
    assert len(sampler) == int((per_bucket // 4).sum())        # only partial batches are dropped


def test_group_sampler_order_is_set_by_the_epoch() -> None:
    """As for a ``DistributedSampler``, the order is a function of (seed, epoch): every rank builds the same list, and
    a resumed run replays it. Lightning calls ``set_epoch`` every epoch."""
    lengths = torch.randint(4, 9, (200,), generator=torch.Generator().manual_seed(0))
    sampler = LengthBucketBatchSampler(lengths, (4, 6, 8), bs=4, seed=0)
    sampler.set_epoch(0)
    first = list(sampler)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_group_sampler_shards_ranks_with_one_bucket_per_step() -> None:
    """Each rank gets the same number of batches, no event twice across ranks, and in every
    step all ranks hold a batch from the same bucket, so no rank waits on a longer one."""
    lengths = torch.randint(4, 9, (403,), generator=torch.Generator().manual_seed(0))
    buckets, bs, world = (4, 6, 8), 4, 4
    samplers = [LengthBucketBatchSampler(lengths, buckets, bs=bs, seed=7, rank=r, world_size=world)
                for r in range(world)]
    for s in samplers:
        s.set_epoch(3)
    per_rank = [list(s) for s in samplers]
    assert len({len(b) for b in per_rank}) == 1 and len(per_rank[0]) == len(samplers[0]) > 0
    seen = torch.tensor([i for b in per_rank for batch in b for i in batch])
    assert seen.unique().numel() == seen.numel()
    bucket_of = torch.bucketize(lengths, torch.tensor(buckets))
    for step in zip(*per_rank):
        assert all(len(batch) == bs for batch in step)
        assert bucket_of[torch.tensor(step)].unique().numel() == 1
    per_bucket = torch.bincount(bucket_of, minlength=3)
    assert len(samplers[0]) == int((per_bucket // (bs * world)).sum())


def test_group_sampler_is_already_distributed_for_lightning() -> None:
    """Lightning wraps any sampler that is not a ``DistributedSampler`` in its own wrapper, which would reshuffle the
    batches across the ranks: the ranks would no longer hold one bucket per step."""
    assert isinstance(LengthBucketBatchSampler(torch.full((8,), 4), (4, 8), bs=4), DistributedSampler)


# ---------------------------------------------------------------- generation

@pytest.mark.parametrize("timesteps", [False, True], ids=["final", "intermediates"])
def test_bucketed_generation_matches_unbucketed(timesteps: bool) -> None:
    ode = {"method": "euler", "step_size": 0.25, "return_timesteps": timesteps}
    plain = _model(_config(odeint=dict(ode)))
    bucketed = _model(_config(odeint=dict(ode, len_buckets=[4, 6])))
    ds = _ragged_batch()                       # lengths 4,5,8,6,4,5 -> buckets 4, 6, 8
    seen = {id(plain): set(), id(bucketed): set()}
    for m in (plain, bucketed):
        m.model.register_forward_pre_hook(lambda mod, args, key=id(m): seen[key].add(args[0].shape[1]))
    torch.manual_seed(2)
    out_p = plain(ds)
    torch.manual_seed(2)
    out_b = bucketed(ds)
    assert seen[id(plain)] == {L_MAX} and seen[id(bucketed)] == {4, 6, 8}
    for a, b in zip(out_b, out_p):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_cheapest_groups_minimise_tokens_plus_a_cost_per_solve() -> None:
    from itertools import combinations
    from legofmt.cfm.solvers import cheapest_groups
    buckets = torch.tensor([8, 10, 12, 14, 16, 20])
    assert cheapest_groups(torch.tensor([100, 100]), buckets[:2], 0).tolist() == [0, 1]
    assert cheapest_groups(torch.tensor([100, 100]), buckets[:2], 10 ** 6).tolist() == [1]
    assert cheapest_groups(torch.tensor([1000, 1, 1000]), buckets[:3], 1000).tolist() == [0, 2]
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):                                 # against every grouping of 6 buckets
        counts = torch.randint(1, 400, (6,), generator=gen)
        cum = [0] + counts.cumsum(0).tolist()
        def cost(tops):
            return sum(500 + int(buckets[j]) * (cum[j + 1] - cum[i]) for i, j in zip([0] + [t + 1 for t in tops], tops))
        every = [list(c) + [5] for r in range(6) for c in combinations(range(5), r)]
        assert cost(cheapest_groups(counts, buckets, 500).tolist()) == min(map(cost, every))


@pytest.mark.parametrize("odeint, n_out, lengths", [
    ({}, N_OUT, {8}),                                   # all events fit the bucket of 8
    ({"len_buckets": None}, N_OUT, {16}),
    ({}, N_OUT + (13,), {8, 16}),                       # 6 events at 8 instead of 16 tokens: worth a solve
    ({}, (1, 2, 13), {16}),                             # 2 events: not worth one, they run with the long event
    ({"len_buckets": [8]}, (1, 2, 13), {8, 16}),        # a list is taken as given
], ids=["default", "off", "worth_a_solve", "not_worth_a_solve", "list"])
def test_generation_buckets_by_length_unless_switched_off(monkeypatch, odeint, n_out, lengths) -> None:
    """Without ``odeint_conf.len_buckets`` generation uses ``auto``: unbucketed, 85 % of the tokens the field runs are
    padding."""
    import legofmt.cfm.solvers as solvers
    monkeypatch.setattr(solvers, "SOLVE_COST", 32, raising=False)
    cfg = _tiny_config({"max_seq_l": 16, "ntypes": 16, "step_cond": False})
    cfg["config"]["odeint_conf"] = {"method": "euler", "step_size": 0.25, **odeint}
    model = _model(cfg)
    seen = set()
    model.model.register_forward_pre_hook(lambda mod, args: seen.add(args[0].shape[1]))
    out = model(_ragged_batch(n_out, L=16))
    assert seen == lengths and out[0].shape[:2] == (len(n_out), 16)


def test_generation_buckets_set_after_construction_take_effect() -> None:
    """lego_eval's per-model ``odeint`` override writes into ``rc.odeint_conf`` after the model is built; a
    ``len_buckets`` set there must bucket generation too (a slab chain pads every call to ``max_seq_l`` = 134 for ~7
    real tokens)."""
    model = _model(_config(odeint={"method": "euler", "step_size": 0.25}))
    model.rc.odeint_conf["len_buckets"] = [4, 6]
    seen = set()
    model.model.register_forward_pre_hook(lambda mod, args: seen.add(args[0].shape[1]))
    model(_ragged_batch())
    assert seen == {4, 6, 8}


def test_compile_limit_counts_buckets_set_after_construction(monkeypatch) -> None:
    """In a static compile every pair (length bucket, ``compile_buckets`` batch size) is its own graph; past dynamo's
    ``recompile_limit`` the solve silently runs eager. The limit must also be raised for buckets that lego_eval's
    ``odeint`` override set after construction."""
    static = dict(fwd_compile=True, compile_dynamic=False, compile_buckets=[2, 4, 8])
    model = _model(_config(odeint={"method": "euler", "step_size": 0.25}))
    model.rc.odeint_conf.update(len_buckets=[4, 6], **static)
    monkeypatch.setattr(torch, "compile", lambda m, **k: m)
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    model(_ragged_batch())
    assert torch._dynamo.config.recompile_limit >= 3 * 3 + 8
    other = _model(_config(odeint={"method": "euler", "step_size": 0.25}))     # a second model adds to the same limit
    other.rc.odeint_conf.update(len_buckets=[4], **static)
    other(_ragged_batch())
    assert torch._dynamo.config.recompile_limit >= 3 * 3 + 2 * 3 + 8


@pytest.mark.parametrize("counts, need, to", [
    ([10, 10, 3, 1], 4, [0, 1, 3, 3]),      # the longest bucket takes the one below it
    ([1, 10, 10], 4, [1, 1, 2]),            # the shortest has none below: it joins the one above
    ([2, 0, 5, 1, 0], 4, [3, 1, 3, 3, 4]),  # a group is named after its longest bucket in use
    ([1, 1, 1], 4, [2, 2, 2]),              # too few events for any bucket: all in one group
    ([3, 0, 1], 0, [0, 1, 2]),              # nothing needed: buckets as given
], ids=["long_end", "short_end", "empty", "all_sparse", "off"])
def test_merge_sparse_fills_every_bucket_in_use(counts, need, to) -> None:
    counts = torch.tensor(counts)
    out = merge_sparse(counts, torch.full_like(counts, need))
    assert out.tolist() == to
    held = torch.zeros_like(counts).index_add_(0, out, counts)
    assert need == 0 or counts.sum() < need or bool((held[held > 0] >= need).all())


def test_group_sampler_merges_buckets_too_sparse_for_min_batches() -> None:
    """A bucket with fewer than ``min_batches`` batches is merged into a longer one. With fine buckets the few longest
    events do not fill a batch of their own; skipped, they would never be trained on."""
    lengths = torch.tensor([4] * 16 + [6] * 9 + [8] * 3)
    sampler = LengthBucketBatchSampler(lengths, (4, 6, 8), bs=4, seed=0, min_batches=2)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 4 + 3                     # bucket 4 alone; 6 and 8 as one
    assert sorted(i for b in batches for i in b) == list(range(28))  # every event shown
    assert all(len({int(lengths[i]) <= 4 for i in b}) == 1 for b in batches)


@pytest.mark.parametrize("spec, counts", [("auto", [200, 0, 0, 0, 203]), ([8, 10, 12, 14], [200, 200, 0, 0, 3])],
                         ids=["auto", "list"])
def test_auto_buckets_are_fitted_to_the_dataset(spec, counts) -> None:
    """``auto`` merges the buckets the dataset leaves sparse; a list is taken as given."""
    cfg = _tiny_config({"max_seq_l": 16, "ntypes": 16, "step_cond": False})
    cfg["config"]["dl_conf"].update(bs=6, len_buckets=spec, len_mode="group")
    model = LEGOLtng(cfg)
    model._train_ds = SimpleNamespace(lengths=torch.tensor([8] * 200 + [10] * 200 + [16] * 3))
    assert model._bucket_sampler().counts.tolist() == counts


def test_group_sampler_lists_one_batch_at_a_time() -> None:
    """The sampler yields an epoch's batches one at a time: listed up front, an epoch is 16M Python ints for the Ar
    cube."""
    import inspect
    sampler = LengthBucketBatchSampler(torch.tensor([4] * 8 + [8] * 8), (4, 8), bs=4, seed=0)
    assert inspect.isgenerator(iter(sampler))
    batches = list(sampler)
    assert all(isinstance(b, list) for b in batches) and sorted(i for b in batches for i in b) == list(range(16))


def test_group_sampler_skips_a_bucket_smaller_than_a_batch() -> None:
    lengths = torch.tensor([4] * 9 + [8] * 3)           # bucket 8 holds fewer than bs events
    sampler = LengthBucketBatchSampler(lengths, (4, 8), bs=4, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 2
    assert all(len(b) == 4 for b in batches)


def test_group_sampler_sizes_batches_by_token_budget() -> None:
    """With ``max_tokens`` a bucket's batch holds ``max_tokens // length`` events, on every rank alike: fewer for a
    long bucket (8: 16 // 8 = 2), more than ``bs`` for a short one (4: 16 // 4 = 4 > ``bs`` 2)."""
    lengths = torch.randint(4, 9, (403,), generator=torch.Generator().manual_seed(0))
    buckets, bs, world = (4, 8), 2, 2
    samplers = [LengthBucketBatchSampler(lengths, buckets, bs=bs, seed=0, rank=r, world_size=world, max_tokens=16)
                for r in range(world)]
    per_rank = [list(s) for s in samplers]
    bucket_of = torch.bucketize(lengths, torch.tensor(buckets))
    sizes = {}
    for step in zip(*per_rank):
        k = int(bucket_of[step[0][0]])
        assert all(len(batch) == len(step[0]) and int(bucket_of[batch].unique()) == k for batch in step)
        assert len(step[0]) * buckets[k] <= 16
        sizes.setdefault(k, set()).add(len(step[0]))
    assert sizes == {0: {4}, 1: {2}}
    per_bucket = torch.bincount(bucket_of, minlength=2)
    assert len(samplers[0]) == len(per_rank[0]) == int(per_bucket[0] // (4 * world) + per_bucket[1] // (2 * world))


def test_transfer_hook_trims_the_batch_to_its_bucket() -> None:
    """The batch crosses to the GPU at its bucket's length, not at ``max_seq_l``: events of length 4 under the buckets
    (4, 8) ship 4 tokens, identical to the first 4 of the padded batch."""
    model = _model(_config(dl={"len_buckets": [4, 8], "len_mode": "group"}))
    ds = _ragged_batch(n_out=(1, 1))
    out = model.on_before_batch_transfer(ds, 0)
    assert out.f.full.shape[1] == 4
    for a, b in ((out.f, ds.f), (out.m, ds.m), (out.am, ds.am)):
        assert torch.equal(a.full, b.full[:, :4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs a GPU")
def test_pinned_batch_is_contiguous_and_unchanged() -> None:
    """``DataStruct.pin_memory`` lets ``DataLoader(pin_memory=True)`` pin a batch. A trimmed (strided) batch comes back
    contiguous: a strided pinned source would be copied through pageable memory."""
    ds = _ragged_batch().trim(5)
    ds.edep_layers = torch.rand(len(ds), 3)
    p = ds.pin_memory()
    for a, b in ((p.f.full, ds.f.full), (p.m.full, ds.m.full), (p.am.full, ds.am.full),
                 (p.edep_layers, ds.edep_layers)):
        assert a.is_pinned() and a.is_contiguous() and torch.equal(a, b)


@pytest.mark.parametrize("dl", [{"len_buckets": "auto", "len_mode": "group", "len_max_tokens": 64}, None,
                                {"len_buckets": "auto", "len_mode": "split", "len_split": 2}],
                         ids=["group", "off", "split"])
def test_schedule_spans_the_real_step_count_in_every_mode(dl) -> None:
    """``configure_optimizers`` rebuilds the schedule over Lightning's step estimate and applies the warmup to the base
    LR once. Group mode's step count is not ``dataset_size / bs``, and ``run.dataset_size`` is hand-written in every
    mode (it must be the post-filter count); a schedule shorter than the run would walk the cosine LR back up."""
    cfg = _config(dl=dl)
    cfg["config"]["opt_conf"] = _muon_cosine()
    model = LEGOLtng(cfg)
    model._trainer = SimpleNamespace(estimated_stepping_batches=37)
    sched = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert sched._milestones == [3] and sched._schedulers[1].T_max == 34
    assert model.rc.opt_conf["scheduler"]["total_steps"] == 37
    assert model.opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 1e-3)


def test_mult_schedule_spans_the_real_step_count() -> None:
    cfg = _mult_config()
    cfg["opt_conf"] = _muon_cosine()
    model = MultModel({"state_dict": {}, "config": cfg})
    model._trainer = SimpleNamespace(estimated_stepping_batches=37)
    assert model.configure_optimizers()["lr_scheduler"]["scheduler"]._schedulers[1].T_max == 34


def test_config_resolves_len_max_tokens() -> None:
    assert resolve_legoltng_config(_config()).len_max_tokens is None
    rc = resolve_legoltng_config(_config(dl={"len_buckets": "auto", "len_mode": "group", "len_max_tokens": 64}))
    assert rc.len_max_tokens == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host-device syncs only exist with a GPU")
def test_training_step_has_no_host_device_sync() -> None:
    """Every sync makes the CPU wait for the GPU, so Python, kernel launches and data loading never overlap GPU work.
    A group-mode step (OT coupling, learned loss weights, flow-map term) has none once Lightning's
    ``on_before_batch_transfer`` has seen the batch on the CPU. The field is a stand-in: x-transformers' eager einx
    ops sync on every call but compile to constants under the ``torch.compile`` that training uses, so they would
    only hide the regressions this test is for."""
    pytest.importorskip("torch_linear_assignment")

    class _VF(torch.nn.Module):
        step_cond = True

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(7, 7)

        def forward(self, x, mask, attn_mask, types, pdgids, t=None, d=None):
            return self.lin(x) * (t + d).unsqueeze(-1)

    model = _model(_config(dl={"len_buckets": [4, 6], "len_mode": "group"}, step_cond=True, ot_coupling=True,
                           learned_loss_weights=True, edep_cell=True, one_step_euler_fac=0.5,
                           one_step_euler_sections=4))
    model.model.vf = _VF()
    model = model.cuda()
    model.train()
    cpu = _ragged_batch(edep_generated=True)
    for sync_check in (False, True):                          # the first step does the lazy setup
        batch = model.on_before_batch_transfer(cpu, 0).to("cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error" if sync_check else 0)
        try:
            model._step(batch, 0).backward()
        finally:
            torch.cuda.set_sync_debug_mode(0)
