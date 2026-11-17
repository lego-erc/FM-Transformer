"""Reading shower data: the cut on raw particles, the Dataset, and the DataLoader factory.

    GetLEGOData                raw ``{per_particle, per_event}`` -> padded particles above ``cutoff_mev``
    LEGODataset                a prepped dataset (a packed or padded file, tensors in memory), indexed by batches
    LengthBucketBatchSampler   batches of events of similar length
    make_loader                the worker plumbing both LightningModules share
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
    """Cuts a raw dataset at ``cutoff_mev`` and pads it: the step before ``DataPrep.prep``.

    Args:
        cutoff_mev:    a particle below it is dropped.
        min_particles: an event left with fewer outgoing particles is dropped.
        device:        where the result goes.
        energy_kin:    cut on the kinetic energy; false (legacy): on |p|.
        dtype:         of the particles (default float32).
    """

    def __init__(self, cutoff_mev=10.0, min_particles=0, device="cpu", energy_kin=True, **kwargs):
        self.dev = device
        self.dtype = kwargs.pop("dtype", torch.float32)
        self.min_particles = min_particles
        self.cutoff_mev = cutoff_mev
        self.energy_kin = energy_kin

    def __call__(self, *args, **kwargs):
        return self.dataset_cutoff(*args, **kwargs)

    def dataset_compact(self, data: dict) -> tuple[Tensor, dict]:
        """Concatenates the incoming and outgoing per-particle tensors into one padded token axis.

        Returns:
            The (N, 1 + n_out, 8) particles and the per-event scalars.
        """
        pp = data.get("per_particle")
        return torch.cat((pp["Incoming"], pp["Outgoing"]), dim=-2).to(self.dtype), data.get("per_event")

    def dataset_cutoff(self, data: dict, n_events: (int | None) = None) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Drops the particles below ``cutoff_mev`` and the events left with fewer than ``min_particles`` outgoing.

        The cut applies to the incoming particle too. Survivors are compacted to the front; padding is NaN.

        Args:
            data:     ``{per_particle: {Incoming, Outgoing}, per_event}``, a particle as
                ``[E_kin, p (3), x (3), pdgid]``.
            n_events: keep a fixed random subset of this many events.

        Returns:
            ``(particles, mask, attn_mask, per_event)``: (N, 1 + n_out, 8) with the incoming particle first, ``mask``
            1 on the real outgoing particles, ``attn_mask`` True on every real particle.
        """
        dataset, data_add = self.dataset_compact(data)
        e_valid = dataset[..., 0] if self.energy_kin else dataset[..., 1:4].norm(dim=-1)   # energy_kin false: legacy
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
    """A prepped dataset, indexed by a batch of event indices at a time (a ``DataStruct`` per call).

    A packed file stays packed: a batch is padded when it is read, and the per-event accessors (``lengths``,
    ``n_out``, ``edep``, ...) answer without padding anything. ``data`` materialises the whole padded dataset.

    Args:
        data: a dataset directory (its packed file if it has one, else the padded one) or a ``.pt`` file, stored in
            MeV; the padded tensors ``(f, m, am[, edep_layers])`` in memory; or a raw dict, which is cut
            (``GetLEGOData``, taking ``kwargs``) and prepped here.
        prep: the ``DataPrep`` whose ``norm_e`` puts the stored channels on the model scale; None: read as stored.
        frac: keep a fixed random fraction of the events.
    """

    def __init__(self, data: (str | dict | tuple), *, prep=None, **kwargs) -> None:
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.packed: PackedData | None = None   # a packed file: batches unpacked on demand
        self._data: DataStruct | None = None
        loaded = isinstance(data, str)  # a fresh load nobody else holds: normalise it in place
        if loaded:
            path = data if data.endswith(".pt") else _dataset_file(data)
            data = torch.load(path, map_location="cpu", weights_only=False)

        if is_packed(data):            # a packed file
            if not loaded:
                raise TypeError("a packed dataset is read from its file: pass the path")
            if prep is not None and not hasattr(prep, "norm_e"):
                raise TypeError("a packed dataset takes a prep with norm_e (a DataPrep) or none")
            self.packed = PackedData.from_dict(data)
            if prep is not None:
                self.packed.norm_e(prep)
        elif isinstance(data, dict):   # raw particles
            if prep is None:
                raise ValueError(
                    "LEGODataset(dict, ...) requires `prep` (e.g. DataPrep(config)); "
                    "GetLEGOData yields pre-format_add layout that DataStruct misaligns."
                )
            data = prep(GetLEGOData(**kwargs)(data))
        elif prep is not None:         # padded tensors, from a file or in memory
            data = prep.norm_e(data, inplace=loaded) if hasattr(prep, "norm_e") else prep(data)
        if self.packed is None:
            self._data = DataStruct(*data)

        frac = kwargs.get("frac")
        if frac and frac < 1.0:   # the same events every run
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
            gb = len(self.packed) * self.packed.L * 41 / 1e9   # bytes per padded token: 8 * 4 + 8 + 1
            warnings.warn(f"materialising the packed dataset as padded tensors (~{gb:.1f} GB); "
                          "index it or use its accessors instead", UserWarning, stacklevel=2)
            self._data = DataStruct(*self.packed.unpack())
        return self._data

    def __len__(self) -> int:
        return len(self._data) if self.packed is None else len(self.packed)

    def __getitem__(self, idx: int | Tensor) -> DataStruct:
        return self._data[idx] if self.packed is None else DataStruct(*self.packed.unpack(idx))

    # ---- per-event fields, without padded tensors on a packed file ---------------------------------------------

    def lengths(self) -> Tensor:
        """Returns the tokens per event (prefix + incoming + outgoing), (N,)."""
        return self._data.am.full.sum(1) if self.packed is None else self.packed.lengths

    def n_out(self) -> Tensor:
        """Returns the outgoing particles per event, (N,)."""
        return self._data.am.out_p.sum(-1) if self.packed is None else self.packed.n_out

    def edep(self) -> Tensor:
        """Returns the E_dep token's value per event, (N,)."""
        return self._data.f.edep if self.packed is None else self.packed.prefix[:, 1]

    def cond(self, name: str) -> Tensor:
        """Returns a conditioning scalar per event, (N,)."""
        return self._data.f.cond(name) if self.packed is None else self.packed.prefix[:, cond_slot(name)]

    def incoming(self) -> Tensor:
        """Returns the incoming particle's token per event, (N, 8)."""
        return self._data.f.in_p[..., 0, :] if self.packed is None else self.packed.incoming

    def species_counts(self, ptypes: Tensor) -> Tensor:
        """Counts the outgoing particles of each species in ``ptypes`` per event, (N, len(ptypes))."""
        if self.packed is not None:
            return self.packed.species_counts(ptypes)
        pid = self._data.f.out_p[..., -1]  # one species at a time: the broadcast compare made an (N, max_out, S) int64
        return torch.stack([(pid == p).sum(1) for p in ptypes], -1)

    def edep_layers(self) -> Tensor | None:
        """Returns the per-layer E_dep, (N, n_layers), or None when the dataset has none."""
        return self._data.edep_layers if self.packed is None else self.packed.edep_layers


def _dataset_file(directory: str) -> str:
    """Returns the directory's packed file if it has one, else its padded one.

    Raises:
        ValueError: the packed file is stale: it sits next to a padded one whose meta.json lacks ``"format": "packed"``
            (a padded writer ran since).
    """
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
        b, idx = self.bucket_of[perm].sort(stable=True)   # the shuffled events, bucket by bucket
        perm = perm[idx]
        counts = torch.bincount(b, minlength=len(self.bs_of))
        batches = []   # this rank's batch of each world_size consecutive ones
        for start, c, bs in zip((counts.cumsum(0) - counts).tolist(), counts.tolist(), self.bs_of.tolist()):
            n = c - c % (bs * self.world_size)
            batches += perm[start:start + n].view(-1, self.world_size, bs)[:, self.rank].tolist()
        return iter([batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()])


def loader_seed(dl_conf: dict) -> int:
    """Returns the shuffle seed, identical on every DDP rank.

    ``dl_conf.seed`` (scripts/train.py sets it from ``run.data_seed``), else Lightning's global seed, else 0.
    """
    return int(dl_conf.get("seed", os.environ.get("PL_GLOBAL_SEED", 0)))


def make_loader(dataset, *, bs, shuffle, num_workers=4, batched_sampler=False, batch_sampler=None, seed=0):
    """Constructs a DataLoader with the worker plumbing both LightningModules share.

    Workers only move CPU-side indexing off the training loop. The shuffle draws from its own ``seed``-ed generator:
    under DDP, Lightning's ``DistributedSamplerWrapper`` hands each rank every W-th batch of the list that rank built,
    so per-rank RNG states (unseeded runs, or rank 0's base-head pretraining) gave overlapping shards -- a quarter of
    every epoch never seen on 2 GPUs.

    Args:
        dataset:         what to read.
        bs:              events per batch.
        shuffle:         shuffle the events each epoch.
        num_workers:     worker processes.
        batched_sampler: hand the dataset a list of indices per call (``batch_size=None``, no default collate), which is
            how ``LEGODataset`` is read; false: torch's per-item fetch and collate.
        batch_sampler:   the sampler of index lists, e.g. a ``LengthBucketBatchSampler``; None: plain batches of ``bs``.
        seed:            the shuffle's seed.
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
