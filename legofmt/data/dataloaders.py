"""Reading prepped shower data: the Dataset objects and the DataLoader factory.

``GetLEGOData`` turns a raw per-particle/per-event dict into the padded
``(features, mask, attn_mask)`` triple, dropping particles below ``cutoff_mev``
and events left with too few survivors. ``LEGODataset`` serves an already-prepped
tensor; ``make_loader`` is the worker plumbing both LightningModules share.
"""

import json
import os
import warnings

import torch
from torch import Tensor
from torch.utils.data import (
    BatchSampler, DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler,
)

from legofmt.data.packed import PackedData, is_packed
from legofmt.data.struct import DataStruct, cond_slot


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
        data_pp = data.get("per_particle")
        data_add = data.get("per_event")
        data_pp = torch.cat((data_pp["Incoming"], data_pp["Outgoing"]), dim=-2)
        return data_pp.to(self.dtype), data_add

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
            rd_idx = torch.randperm(data_pp.shape[0], device="cpu", generator=torch.Generator().manual_seed(0))[:n_events]
            data_pp = data_pp[rd_idx]
            data_add = {k: v[rd_idx] for k, v in data_add.items()}
        particle_nan = ~data_pp.isnan().any(dim=-1)
        attn_mask = particle_nan.to(torch.int64)
        mask = attn_mask.clone()
        mask[:, 0] = 0
        data_add = {k: v.to(self.dev) for k, v in data_add.items()}
        return data_pp.to(self.dev), mask.to(self.dev), attn_mask.to(self.dev).bool(), data_add


class LEGODataset(Dataset):
    def __init__(self, data: (str | dict | tuple), *, prep=None, **kwargs) -> None:
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.packed: PackedData | None = None   # a packed file: batches unpacked on demand
        self._data: DataStruct | None = None
        loaded = isinstance(data, str)  # a fresh load nobody else holds: normalise it in place
        if loaded:
            path = data if data.endswith(".pt") else _dataset_file(data)
            data = torch.load(path, map_location="cpu", weights_only=False)
        if is_packed(data):
            if not loaded:
                raise TypeError("a packed dataset is read from its file: pass the path")
            if prep is not None and not hasattr(prep, "norm_e"):
                raise TypeError("a packed dataset takes a prep with norm_e (a DataPrep) or none")
            self.packed = PackedData.from_dict(data)
            if prep is not None:
                self.packed.norm_e(prep)
        elif isinstance(data, dict):
            if prep is None:
                raise ValueError(
                    "LEGODataset(dict, ...) requires `prep` (e.g. DataPrep(config)); "
                    "GetLEGOData yields pre-format_add layout that DataStruct misaligns."
                )
            data = prep(GetLEGOData(**kwargs)(data))
        elif prep is not None:
            data = prep.norm_e(data, inplace=loaded) if hasattr(prep, "norm_e") else prep(data)
        if self.packed is None:
            self._data = DataStruct(*data)

        frac = kwargs.get("frac")
        if frac and frac < 1.0:
            n = int(len(self) * frac)
            idxs = torch.randperm(len(self), device=self.device,
                                  generator=torch.Generator(device=self.device).manual_seed(0))[:n]
            if self.packed is None:
                self._data = self._data[idxs]
            else:
                self.packed = self.packed.select(idxs.cpu())

    @property
    def data(self) -> DataStruct:
        """The whole dataset as padded tensors; a packed file materialises them here, once."""
        if self._data is None:
            gb = len(self.packed) * self.packed.L * 41 / 1e9   # bytes per padded row: 8 * 4 + 8 + 1
            warnings.warn(f"materialising the packed dataset as padded tensors (~{gb:.1f} GB); "
                          "index it or use its accessors instead", UserWarning, stacklevel=2)
            self._data = DataStruct(*self.packed.unpack())
        return self._data

    def __len__(self) -> int:
        return len(self._data) if self.packed is None else len(self.packed)

    def __getitem__(self, idx: int | Tensor) -> DataStruct:
        return self._data[idx] if self.packed is None else DataStruct(*self.packed.unpack(idx))

    # Per-event fields, without padded tensors on a packed file.
    def lengths(self) -> Tensor:
        return self._data.am.full.sum(1) if self.packed is None else self.packed.lengths

    def n_out(self) -> Tensor:
        return self._data.am.out_p.sum(-1) if self.packed is None else self.packed.n_out

    def edep(self) -> Tensor:
        return self._data.f.edep if self.packed is None else self.packed.prefix[:, 1]

    def cond(self, name: str) -> Tensor:
        return self._data.f.cond(name) if self.packed is None else self.packed.prefix[:, cond_slot(name)]

    def incoming(self) -> Tensor:
        return self._data.f.in_p[..., 0, :] if self.packed is None else self.packed.incoming

    def species_counts(self, ptypes: Tensor) -> Tensor:
        if self.packed is not None:
            return self.packed.species_counts(ptypes)
        pid = self._data.f.out_p[..., -1]  # one species at a time: the broadcast compare made an (N, max_out, S) int64
        return torch.stack([(pid == p).sum(1) for p in ptypes], -1)

    def edep_layers(self) -> Tensor | None:
        return self._data.edep_layers if self.packed is None else self.packed.edep_layers


def _dataset_file(directory: str) -> str:
    """The directory's packed file if it has one, else its padded one. A packed file next to a
    padded one whose meta.json lacks ``"format": "packed"`` is stale (a padded writer ran since)."""
    packed, padded = directory + "/data_packed.pt", directory + "/data_prepped.pt"
    if not os.path.exists(packed):
        return padded
    meta = directory + "/meta.json"
    if os.path.exists(padded) and os.path.exists(meta):
        with open(meta) as fh:
            if json.load(fh).get("format") != "packed":
                raise ValueError(f"{directory}: data_packed.pt is stale (meta.json was written for data_prepped.pt); "
                                 "delete it or pass the file explicitly")
    return packed


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
        return DataLoader(
            dataset,
            sampler=batch_sampler,
            batch_size=None,
            **common,
        )
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, **common)
