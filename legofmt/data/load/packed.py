"""A dataset without its padding (each event's real tokens plus where they start), and reading it from its file.

    load_dataset   a dataset directory or file -> PackedData, on the model scale
    PackedData     the events; ``dataset[idx]`` pads a batch of them to a ``DataStruct``

Everything but the particle tokens of a padded ``(f, m, am[, edep_layers])`` follows from a few per-event numbers:

```
prefix tokens   [prefix[i, j], *prefix_token[1:]]
am              one block of n_prefix + 1 + n_out tokens from token 0
m               am, with its first n_prefix + 1 entries [0, 1, 0, ...]
padded tokens   [pad_e0[i], *pad_token[1:]]
```

Those values are copied when packing, so ``unpack`` gives the padded tensors back bit for bit; the packing
(``legofmt.data.write.pack``) refuses anything else.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, fields
from functools import cached_property

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.data.struct import DataStruct, cond_slot
from legofmt.legacy.dataset import PaddedData

PACKED_FORMAT = "lego-packed"
PACKED_VERSION = 1


def is_packed(obj) -> bool:
    """Returns whether ``obj`` is a packed dataset as ``torch.load`` gives it (``PackedData.to_dict``)."""
    return isinstance(obj, dict) and obj.get("format") == PACKED_FORMAT


def loss_mask(am: Tensor, n_prefix: int) -> Tensor:
    """Builds the loss mask ``m``: ``am`` minus the prefix and incoming tokens, plus the E_dep token (slot 1)."""
    m = am.long()
    m[..., :n_prefix + 1] = 0
    m[..., 1] = 1
    return m


@dataclass
class PackedData:
    """A prepped dataset of N events without its padding; ``unpack`` pads events back to ``L`` tokens.

    Indexed by a batch of event indices at a time, it gives a ``DataStruct``. The per-event accessors (``lengths``,
    ``n_out``, ``edep``, ...) answer without padding anything; ``data`` materialises the whole padded dataset.
    """

    prefix: Tensor         # (N, n_prefix) channel 0 of the prefix tokens
    tokens: Tensor         # (R, 8) every event's incoming and outgoing tokens, back to back
    offsets: Tensor        # (N + 1,) event i's tokens are tokens[offsets[i]:offsets[i + 1]]
    pad_e0: Tensor         # (N,) channel 0 of an event's padded tokens (NaN: none)
    pad_token: Tensor      # (8,) channels 1: of every padded token
    prefix_token: Tensor   # (8,) channels 1: of every prefix token
    n_prefix: int
    L: int
    edep_layers: Tensor | None = None

    def __len__(self) -> int:
        return self.prefix.shape[0]

    def __getitem__(self, idx: int | Tensor) -> DataStruct:
        return DataStruct(*self.unpack(idx))

    @cached_property
    def data(self) -> DataStruct:
        """The whole dataset as padded tensors, materialised here, once."""
        gb = len(self) * self.L * 41 / 1e9   # bytes per padded token: 8 * 4 + 8 + 1
        warnings.warn(f"materialising the packed dataset as padded tensors (~{gb:.1f} GB); "
                      "index it or use its accessors instead", UserWarning, stacklevel=3)
        return DataStruct(*self.unpack())

    # ---- per-event fields, without padding anything ------------------------------------------------------------

    @property
    def lengths(self) -> Tensor:
        """Tokens per event in the padded layout (prefix + incoming + outgoing)."""
        return self.n_prefix + self.offsets.diff()

    @property
    def n_out(self) -> Tensor:
        """Outgoing particles per event."""
        return self.offsets.diff() - 1

    @property
    def edep(self) -> Tensor:
        """The E_dep token's value of every event, (N,)."""
        return self.prefix[:, 1]

    def cond(self, name: str) -> Tensor:
        """Returns a conditioning scalar per event, (N,)."""
        return self.prefix[:, cond_slot(name)]

    @property
    def incoming(self) -> Tensor:
        """The incoming particle's token of every event, (N, 8)."""
        return self.tokens[self.offsets[:-1]]

    def species_counts(self, ptypes: Tensor) -> Tensor:
        """Counts the outgoing particles of each species per event, ``(N, len(ptypes))``."""
        ptypes = torch.as_tensor(ptypes)
        if (ptypes == 0).any():
            raise ValueError("pdgid 0 is the padding's: a padded file would count its padded tokens")
        ev = torch.repeat_interleave(torch.arange(len(self)), self.offsets.diff())
        out = torch.ones(self.tokens.shape[0], dtype=torch.bool)
        out[self.offsets[:-1]] = False
        pid, ev = self.tokens[out, -1], ev[out]
        return torch.stack([torch.bincount(ev[pid == p], minlength=len(self)) for p in ptypes], -1)

    def select(self, idx: Tensor) -> "PackedData":
        """Returns the events ``idx`` (a 1-D index, repeats allowed) as a packed dataset of their own."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        lens = self.offsets[idx + 1] - self.offsets[idx]
        offsets = F.pad(lens.cumsum(0), (1, 0))
        src = torch.repeat_interleave(self.offsets[idx] - offsets[:-1], lens) + torch.arange(int(offsets[-1]))
        return PackedData(self.prefix[idx], self.tokens[src], offsets, self.pad_e0[idx], self.pad_token,
                          self.prefix_token, self.n_prefix, self.L,
                          None if self.edep_layers is None else self.edep_layers[idx])

    def unpack(self, idx=slice(None)) -> tuple:
        """Pads the events ``idx`` to full ``L``.

        Args:
            idx: any index of the event axis; its shape becomes the batch shape.

        Returns:
            ``(f, m, am)`` as they were packed, with ``edep_layers`` as a 4th tensor when the dataset carries it.
        """
        P, L = self.n_prefix, self.L
        start = self.offsets[:-1][idx]
        lens = self.offsets[1:][idx] - start
        f = torch.empty(*lens.shape, L, 8)
        f[..., :P, :] = self.prefix_token
        f[..., :P, 0] = self.prefix[idx]
        f[..., P:, :] = self.pad_token
        f[..., P:, 0] = self.pad_e0[idx][..., None]
        j = torch.arange(L - P)
        valid = j < lens[..., None]
        f[..., P:, :][valid] = self.tokens[(start[..., None] + j)[valid]]
        am = torch.cat((torch.ones(*lens.shape, P, dtype=torch.bool), valid), -1)
        out = (f, loss_mask(am, P), am)
        return out if self.edep_layers is None else out + (self.edep_layers[idx],)

    def norm_e(self, prep) -> None:
        """Applies ``prep.norm_e`` in place to all events at once, as the padded file gets it.

        Only the tokens it touches are built: prefix, incoming, edep_layers.
        """
        N, P = len(self), self.n_prefix
        v = torch.empty(N, P + 1, 8)
        v[:, :P] = self.prefix_token
        v[:, :P, 0] = self.prefix
        v[:, P] = self.incoming
        batch = (v, torch.zeros(N, P + 1, dtype=torch.long), torch.ones(N, P + 1, dtype=torch.bool))
        out = prep.norm_e(batch + (() if self.edep_layers is None else (self.edep_layers,)), inplace=True)
        self.prefix = out[0][:, :P, 0].clone()
        self.tokens[self.offsets[:-1], 0] = out[0][:, P, 0]
        if self.edep_layers is not None:
            self.edep_layers = out[3]

    def to_dict(self) -> dict:
        """Returns the fields plus format and version: what a packed file stores."""
        stored = {f.name: getattr(self, f.name) for f in fields(self)}   # not the cached ``data``
        return {"format": PACKED_FORMAT, "version": PACKED_VERSION, **stored}

    @classmethod
    def from_dict(cls, d: dict) -> "PackedData":
        """Constructs the dataset from a packed file's dict.

        Raises:
            ValueError: not a packed file of this version.
        """
        if not is_packed(d) or d.get("version") != PACKED_VERSION:
            raise ValueError(f"not a {PACKED_FORMAT} v{PACKED_VERSION} file: "
                             f"format={d.get('format')!r} version={d.get('version')!r}")
        return cls(**{k: v for k, v in d.items() if k not in ("format", "version")})


def load_dataset(data: str | dict | tuple, *, prep=None, frac=None, device="cpu", **cut_args) -> PackedData:
    """Loads a dataset file, puts its stored channels on the model scale and keeps a fixed fraction of its events.

    Args:
        data:     a dataset directory (its packed file) or a ``.pt`` file, stored in MeV. Legacy: a padded file, the
            padded tensors in memory, or a raw dict (cut with ``cut_args``).
        prep:     the ``DataPrep`` whose ``norm_e`` puts the stored channels on the model scale; None: read as stored.
        frac:     keep a fixed random fraction of the events.
        device:   of ``frac``'s generator.

    Returns:
        The events; a legacy input gives a ``PaddedData``, which answers the same calls.

    Raises:
        TypeError: a packed dataset not read from its file, or a ``prep`` without ``norm_e``.
    """
    from_file = isinstance(data, str)
    if from_file:
        path = data if data.endswith(".pt") else _dataset_file(data)
        data = torch.load(path, map_location="cpu", weights_only=False)
    if not is_packed(data):   # legacy: padded tensors, raw particles
        return PaddedData.read(data, prep, frac, device, inplace=from_file, **cut_args)

    if not from_file:
        raise TypeError("a packed dataset is read from its file: pass the path")
    if prep is not None and not hasattr(prep, "norm_e"):
        raise TypeError("a packed dataset takes a prep with norm_e (a DataPrep) or none")
    events = PackedData.from_dict(data)
    if prep is not None:
        events.norm_e(prep)
    if frac and frac < 1.0:   # the same events every run
        n = int(len(events) * frac)
        idx = torch.randperm(len(events), device=device, generator=torch.Generator(device=device).manual_seed(0))[:n]
        events = events.select(idx.cpu())
    return events


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
