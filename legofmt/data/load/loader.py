"""Batches of a dataset: the length-bucket sampler and the DataLoader factory.

    LengthBucketBatchSampler   batches of events of similar length
    merge_sparse               the buckets the sampler merges because they hold too few events
    smallest_bucket            the bucket a length falls in
    event_lengths              each event's token count, also through an ``EventView``
    split_val                  the train / validation split, by ``val_conf``
    loader_seed                the shuffle seed, the same on every DDP rank
    make_loader                the DataLoader of both LightningModules
"""

import os

import torch
from torch import Tensor
from torch.utils.data import (
    BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler,
)

from legofmt.data.load.packed import EventView


class LengthBucketBatchSampler(DistributedSampler):
    """A batch sampler whose batches hold events of one length bucket, so that a batch can be trimmed to that length.

    Each epoch the events are shuffled, every bucket is cut into batches, and the batches are shuffled. The events of
    a bucket that do not fill one more batch per DDP rank are left out of that epoch. The order depends on
    ``seed + epoch`` only, so every rank cuts the same batches and takes its own one of every ``world_size``
    consecutive batches. It is a ``DistributedSampler`` so that Lightning does not wrap it.

    Args:
        lengths:     (N,), the tokens per event.
        buckets:     the bucket lengths, ascending; an event goes to the shortest bucket that holds it, and the last
            one must hold the longest event.
        bs:          events per batch.
        seed:        the shuffle's seed.
        rank:        this process's DDP rank.
        world_size:  the number of DDP processes.
        max_tokens:  a bucket of length ``b`` takes ``max_tokens // b`` events per batch (at least one) instead of
            ``bs``.
        min_batches: a bucket with fewer batches per epoch is merged into a longer one (``merge_sparse``). 0: no
            merging; a bucket with too few events for one batch is never sampled.
    """

    def __init__(self, lengths: Tensor, buckets, bs: int, seed: int = 0, rank: int = 0, world_size: int = 1,
                 max_tokens: int | None = None, min_batches: int = 0) -> None:
        buckets = torch.as_tensor(buckets)
        self.bs_of = torch.full_like(buckets, bs) if max_tokens is None else (max_tokens // buckets).clamp(min=1)
        bucket_of = torch.searchsorted(buckets, torch.as_tensor(lengths), out_int32=True)
        merged = merge_sparse(torch.bincount(bucket_of, minlength=len(buckets)), min_batches * self.bs_of * world_size)
        self.bucket_of = merged.int()[bucket_of]
        self.counts = torch.bincount(self.bucket_of, minlength=len(buckets))
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.n_batches = int((self.counts // (self.bs_of * world_size)).sum())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.bucket_of), generator=gen)
        perm = perm[self.bucket_of[perm].argsort(stable=True)]   # the shuffled events, bucket by bucket
        batches = []   # this rank's batch of each world_size consecutive ones
        for events, bs in zip(perm.split(self.counts.tolist()), self.bs_of.tolist()):
            n = len(events) - len(events) % (bs * self.world_size)
            batches += events[:n].unflatten(0, (-1, self.world_size, bs))[:, self.rank].unbind(0)
        return (batches[i].tolist() for i in torch.randperm(len(batches), generator=gen).tolist())


# ``len_buckets: auto``: a bucket with fewer batches per epoch is merged into a longer one (``merge_sparse``)
AUTO_MIN_BATCHES = 16


def merge_sparse(counts: Tensor, need: Tensor) -> Tensor:
    """Returns, for each bucket, the bucket its events are batched in; a bucket in use holds at least its ``need``.

    From the longest bucket down, a bucket takes the events of the shorter ones until it holds its ``need``; the next
    shorter bucket starts a new group. If the shortest group stays below its ``need``, it joins the next longer
    group. Only when all events together stay below the longest bucket's ``need`` does that bucket hold fewer. Without
    the merge, fine buckets leave the longest events short of one batch, and they are never trained on.

    Args:
        counts: (K,), the events per bucket, the buckets ascending.
        need:   (K,), the events a bucket must hold to be used.

    Returns:
        (K,), the index of the bucket that takes each bucket's events: the longest bucket of its group.
    """
    to, top, held, above = torch.arange(len(counts)), None, 0, None
    for k in reversed(range(len(counts))):
        top = k if held == 0 else top   # a group is named after its longest bucket
        to[k], held = top, held + int(counts[k])
        if held >= need[top]:
            above, held = top, 0
    return to if held == 0 or above is None else torch.where(to == top, above, to)


def smallest_bucket(n: Tensor, buckets) -> Tensor:
    """Returns the smallest of the sorted ``buckets`` that holds each length of ``n``."""
    buckets = torch.as_tensor(buckets, device=n.device)
    return buckets[torch.searchsorted(buckets, n.contiguous())]


def event_lengths(ds) -> Tensor:
    """Returns each event's token count, (N,); for an ``EventView``, of the events it holds."""
    return event_lengths(ds.ds)[ds.idx] if isinstance(ds, EventView) else ds.lengths


def split_val(dataset, val_conf: dict):
    """Splits ``dataset`` into ``(train, val)``, two ``EventView``s on it.

    The validation set is a random ``val_conf.val_frac`` of the events (default 0.01, at least one event), picked with
    ``val_conf.seed`` (default 0). Both views hold the events ``torch.utils.data.random_split`` picks, as index
    tensors.
    """
    n_val = max(1, int(len(dataset) * val_conf.get("val_frac", 0.01)))
    gen   = torch.Generator().manual_seed(val_conf.get("seed", 0))
    perm  = torch.randperm(len(dataset), generator=gen)
    return EventView(dataset, perm[:-n_val]), EventView(dataset, perm[-n_val:])


def loader_seed(dl_conf: dict) -> int:
    """Returns the shuffle seed, identical on every DDP rank.

    ``dl_conf.seed`` (scripts/train.py sets it from ``run.data_seed``), else Lightning's global seed, else 0.
    """
    return int(dl_conf.get("seed", os.environ.get("PL_GLOBAL_SEED", 0)))


def make_loader(dataset, *, bs, shuffle, num_workers=4, batched_sampler=True, batch_sampler=None, seed=0):
    """Constructs a DataLoader that hands the dataset a list of event indices per call.

    The workers only move the dataset's indexing off the training loop. The shuffle has its own generator, seeded
    with ``seed``: under DDP, Lightning's ``DistributedSamplerWrapper`` gives each rank every ``world_size``-th batch
    of the list that rank built, so the ranks must build the same list. With the process's RNG, whose state can
    differ between the ranks, their shares overlap and part of every epoch is never seen.

    Args:
        dataset:         a loaded dataset (``load_dataset``), or an ``EventView`` of one.
        bs:              events per batch.
        shuffle:         shuffle the events each epoch.
        num_workers:     worker processes.
        batched_sampler: false (legacy): torch's per-item fetch and collate.
        batch_sampler:   the sampler of index lists, e.g. a ``LengthBucketBatchSampler``, in place of ``bs``,
            ``shuffle`` and ``seed``; None: plain batches of ``bs``.
        seed:            the shuffle's seed.
    """
    workers = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
                   multiprocessing_context="fork" if num_workers > 0 else None)
    if not batched_sampler: return DataLoader(dataset, batch_size=bs, shuffle=shuffle, **workers)   # legacy
    if batch_sampler is None:
        sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)) if shuffle
                   else SequentialSampler(dataset))
        batch_sampler = BatchSampler(sampler, bs, drop_last=False)
    return DataLoader(dataset, sampler=batch_sampler, batch_size=None, **workers)
