"""Batches of a dataset: the length-bucket sampler and the DataLoader factory.

    LengthBucketBatchSampler   batches of events of similar length
    make_loader                the worker plumbing both LightningModules share
"""

import os

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler


class LengthBucketBatchSampler(DistributedSampler):
    """Samples batches whose events share a length bucket, so a batch is padded only to its bucket's length.

    An epoch shuffles the events, cuts every bucket into batches (its incomplete last one is dropped) and shuffles the
    batches. The order is a function of ``seed + epoch`` only: every DDP rank builds the same list and takes its own
    batch of each ``world_size`` consecutive ones. A ``DistributedSampler`` so that Lightning does not wrap it.

    Args:
        lengths:    (N,), the tokens per event.
        buckets:    ascending lengths; an event goes to the first bucket that holds it.
        bs:         events per batch.
        seed:       the shuffle's seed.
        rank:       this DDP process.
        world_size: the number of DDP processes.
        max_tokens: a bucket of length ``b`` takes ``max_tokens // b`` events per batch instead of ``bs``.
    """

    def __init__(self, lengths: Tensor, buckets, bs: int, seed: int = 0,
                 rank: int = 0, world_size: int = 1, max_tokens: int | None = None) -> None:
        buckets = torch.as_tensor(buckets)
        self.bucket_of = torch.bucketize(torch.as_tensor(lengths), buckets)
        self.counts = torch.bincount(self.bucket_of, minlength=len(buckets))
        self.bs_of = torch.full_like(buckets, bs) if max_tokens is None else (max_tokens // buckets).clamp(min=1)
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
            batches += events[:n].view(-1, self.world_size, bs)[:, self.rank].tolist()
        return iter([batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()])


def loader_seed(dl_conf: dict) -> int:
    """Returns the shuffle seed, identical on every DDP rank.

    ``dl_conf.seed`` (scripts/train.py sets it from ``run.data_seed``), else Lightning's global seed, else 0.
    """
    return int(dl_conf.get("seed", os.environ.get("PL_GLOBAL_SEED", 0)))


def make_loader(dataset, *, bs, shuffle, num_workers=4, batched_sampler=True, batch_sampler=None, seed=0):
    """Constructs a DataLoader that hands the dataset a list of event indices per call.

    Workers only move CPU-side indexing off the training loop. The shuffle draws from its own ``seed``-ed generator:
    under DDP, Lightning's ``DistributedSamplerWrapper`` hands each rank every W-th batch of the list that rank built,
    so per-rank RNG states (unseeded runs, or rank 0's base-head pretraining) gave overlapping shards -- a quarter of
    every epoch never seen on 2 GPUs.

    Args:
        dataset:         what to read: a loaded dataset (``load_dataset``), or a subset of one.
        bs:              events per batch.
        shuffle:         shuffle the events each epoch.
        num_workers:     worker processes.
        batched_sampler: false (legacy): torch's per-item fetch and collate.
        batch_sampler:   the sampler of index lists, e.g. a ``LengthBucketBatchSampler``; None: plain batches of ``bs``.
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
