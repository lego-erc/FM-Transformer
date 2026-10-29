"""Reading prepped shower data: the Dataset objects and the DataLoader factory.

``GetLEGOData`` turns a raw per-particle/per-event dict into the padded
``(features, mask, attn_mask)`` triple, dropping particles below ``cutoff_mev``
and events left with too few survivors. ``LEGODataset`` serves an already-prepped
tensor; ``make_loader`` is the worker plumbing both LightningModules share.
"""

import os

import torch
from torch import Tensor
from torch.utils.data import (
    BatchSampler, DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler,
)

from legofmt.data.struct import DataStruct


class GetLEGOData:
    def __init__(
        self,
        cutoff_mev=10.0,
        min_particles=0,
        device="cpu",
        energy_kin=True,
        **kwargs,
    ):
        self.dev = device
        self.dtype = kwargs.pop("dtype", torch.float32)
        self.min_particles = min_particles
        self.cutoff_mev = cutoff_mev
        self.energy_kin = energy_kin

    def __call__(self, *args, **kwargs):
        return self.dataset_cutoff(*args, **kwargs)

    def dataset_compact(self, data):
        """Concatenate the incoming and outgoing per-particle tensors into one padded row axis."""
        pp = data.get("per_particle")
        return torch.cat((pp["Incoming"], pp["Outgoing"]), dim=-2).to(self.dtype), data.get("per_event")

    def dataset_cutoff(
        self,
        data: dict,
        n_events: (int | None) = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Drop particles below cutoff_mev (kinetic energy when energy_kin, else |p|;
        the incoming row included), compact survivors to the front, drop events with
        fewer than min_particles outgoing; padding is NaN."""
        dataset, data_add = self.dataset_compact(data)
        e_valid = dataset[..., 0] if self.energy_kin else dataset[..., 1:4].norm(dim=-1)
        mask_valid = e_valid >= self.cutoff_mev
        max_valid = mask_valid.sum(dim=-1).max()
        mask_valid_sorted = mask_valid.sort(dim=-1, descending=True).values[:, :max_valid]
        dataset_valid = torch.full_like(dataset[:, :max_valid], torch.nan)
        dataset_valid[mask_valid_sorted] = dataset[mask_valid]
        del dataset, mask_valid, mask_valid_sorted
        keep_complete = ~dataset_valid[:, : self.min_particles + 1, 0].isnan().any(dim=-1)
        data_pp = dataset_valid[keep_complete]
        del dataset_valid
        data_add = {k: v[keep_complete] for k, v in data_add.items()}
        if n_events is not None:
            rd_idx = torch.randperm(data_pp.shape[0], generator=torch.Generator().manual_seed(0))[:n_events]
            data_pp = data_pp[rd_idx]
            data_add = {k: v[rd_idx] for k, v in data_add.items()}
        attn_mask = (~data_pp.isnan().any(dim=-1)).to(torch.int64)
        mask = attn_mask.clone()
        mask[:, 0] = 0
        data_add = {k: v.to(self.dev) for k, v in data_add.items()}
        return data_pp.to(self.dev), mask.to(self.dev), attn_mask.to(self.dev).bool(), data_add


class LEGODataset(Dataset):
    def __init__(self, data: (str | dict | tuple), *, prep=None, **kwargs) -> None:
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        loaded = isinstance(data, str)  # a fresh load nobody else holds: normalise it in place
        if loaded:
            path = data if data.endswith(".pt") else data + "/data_prepped.pt"
            data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if prep is None:
                raise ValueError(
                    "LEGODataset(dict, ...) requires `prep` (e.g. DataPrep(config)); "
                    "GetLEGOData yields pre-format_add layout that DataStruct misaligns."
                )
            data = prep(GetLEGOData(**kwargs)(data))
        elif prep is not None:
            data = prep.norm_e(data, inplace=loaded) if hasattr(prep, "norm_e") else prep(data)
        self.data = DataStruct(*data)

        frac = kwargs.get("frac")
        if frac and frac < 1.0:
            n = int(len(self.data) * frac)
            idxs = torch.randperm(len(self.data), device=self.device,
                                  generator=torch.Generator(device=self.device).manual_seed(0))[:n]
            self.data = self.data[idxs]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | Tensor) -> DataStruct:
        return self.data[idx]


class LengthBucketBatchSampler(DistributedSampler):
    def __init__(self, lengths: Tensor, buckets, bs: int, seed: int = 0,
                 rank: int = 0, world_size: int = 1, max_tokens: int | None = None) -> None:
        buckets = torch.as_tensor(buckets)
        self.bucket_of = torch.bucketize(torch.as_tensor(lengths), buckets)
        self.bs_of = torch.full_like(buckets, bs) if max_tokens is None else (max_tokens // buckets).clamp(min=1)
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        counts = torch.bincount(self.bucket_of, minlength=len(buckets))
        self.n_batches = int((counts // (self.bs_of * world_size)).sum())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.bucket_of), generator=gen)
        b, idx = self.bucket_of[perm].sort(stable=True)
        perm = perm[idx]
        counts = torch.bincount(b, minlength=len(self.bs_of))
        batches = []
        for start, c, bs in zip((counts.cumsum(0) - counts).tolist(), counts.tolist(), self.bs_of.tolist()):
            n = c - c % (bs * self.world_size)
            batches += perm[start:start + n].view(-1, self.world_size, bs)[:, self.rank].tolist()
        return iter([batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()])


def loader_seed(dl_conf: dict) -> int:
    """The shuffle seed, identical on every DDP rank: ``dl_conf.seed`` (scripts/train.py
    sets it from ``run.data_seed``), else Lightning's global seed, else 0."""
    return int(dl_conf.get("seed", os.environ.get("PL_GLOBAL_SEED", 0)))


def make_loader(dataset, *, bs, shuffle, num_workers=4, batched_sampler=False, batch_sampler=None, seed=0):
    """The worker plumbing both LightningModules share.

    ``batched_sampler`` hands the dataset a tensor of indices per call
    (``batch_size=None``, no default collate), which is how ``LEGODataset`` is
    read; the plain path is torch's per-item fetch and collate. Workers only
    move CPU-side indexing off the training loop.

    The shuffle draws from its own ``seed``-ed generator: under DDP, Lightning's
    ``DistributedSamplerWrapper`` hands each rank every W-th batch of the list
    that rank built, so per-rank RNG states (unseeded runs, or rank 0's base-head
    pretraining) gave overlapping shards -- a quarter of every epoch never seen
    on 2 GPUs.
    """
    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    if batched_sampler:
        if batch_sampler is None:
            sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)) if shuffle
                       else SequentialSampler(dataset))
            batch_sampler = BatchSampler(sampler, bs, drop_last=False)
        return DataLoader(dataset, sampler=batch_sampler, batch_size=None, **common)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, **common)
