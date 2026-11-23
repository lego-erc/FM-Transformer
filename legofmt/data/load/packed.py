"""A dataset without its padding (each event's real tokens plus where they start), and reading it from its file.

    load_dataset       a dataset directory or file -> PackedData, on the model scale
    PackedData         the events; ``dataset[idx]`` pads a batch of them to a ``DataStruct``
    EventView          some of a dataset's events, by index: the train / validation split, ``interacting_only``
    interacting_only   the view without the pass-through events (``model_conf.exclude_passthrough``)
    is_packed          whether a loaded object is a packed dataset

Everything in a padded ``(f, m, am)`` but its particle tokens follows from a few numbers per event and two template
tokens:

```
prefix tokens    [prefix[i, j], *prefix_token[1:]]
am               True on the first n_prefix + 1 + n_out tokens
m                am, with its first n_prefix + 1 entries [0, 1, 0, ...]
padding tokens   [pad_e0[i], *pad_token[1:]]
```

Packing (``legofmt.data.write.pack``) copies those values, so ``unpack`` gives the padded tensors back bit for bit;
it refuses tensors for which that would not hold.
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

from legofmt.data.struct import DataStruct, cond_slot, loss_mask
from legofmt.legacy.dataset import PaddedData, token_names

PACKED_FORMAT = "lego-packed"
PACKED_VERSION = 1


def is_packed(obj) -> bool:
    """Returns whether ``obj`` is a packed dataset as ``torch.load`` gives it (``PackedData.to_dict``)."""
    return isinstance(obj, dict) and obj.get("format") == PACKED_FORMAT


@dataclass
class PackedData:
    """A prepped dataset of N events without its padding; ``unpack`` pads events back to ``L`` tokens.

    ``dataset[idx]`` with a batch of event indices gives a ``DataStruct``. The per-event accessors (``lengths``,
    ``n_out``, ``edep``, ...) answer without padding anything; ``data`` materialises the whole padded dataset. The
    incoming energy, E_dep and ``edep_layers`` are in MeV as the file stores them, and on the model scale once
    ``norm_e`` has run.
    """

    prefix: Tensor         # (N, n_prefix) channel 0 of the prefix tokens
    tokens: Tensor         # (R, 8) the incoming and outgoing tokens of all events, back to back
    offsets: Tensor        # (N + 1,) event i's tokens are tokens[offsets[i]:offsets[i + 1]]
    pad_e0: Tensor         # (N,) channel 0 of the event's padding tokens (NaN: the event has none)
    pad_token: Tensor      # (8,) channels 1: of every padding token (channel 0: ``pad_e0``)
    prefix_token: Tensor   # (8,) channels 1: of every prefix token (channel 0: ``prefix``)
    n_prefix: int          # the number of prefix tokens
    L: int                 # the tokens per event in the padded layout
    edep_layers: Tensor | None = None   # (N, n_layers) the per-layer E_dep, when the dataset carries it

    def __len__(self) -> int:
        return self.prefix.shape[0]

    def __getitem__(self, idx: int | Tensor) -> DataStruct:
        return DataStruct(*self.unpack(idx))

    @cached_property
    def data(self) -> DataStruct:
        """The whole dataset as padded tensors, materialised here, once."""
        gb = len(self) * self.L * 41 / 1e9   # bytes per token of the padded tensors: f 8 * 4, m 8, am 1
        warnings.warn(f"materialising the packed dataset as padded tensors (~{gb:.1f} GB); "
                      "index it or use its accessors instead", UserWarning, stacklevel=3)
        return DataStruct(*self.unpack())

    # ---- per-event fields, without padding anything ------------------------------------------------------------

    @property
    def lengths(self) -> Tensor:
        """Tokens per event in the padded layout (prefix + incoming + outgoing), (N,)."""
        return self.n_prefix + self.offsets.diff()

    @property
    def n_out(self) -> Tensor:
        """Outgoing particles per event, (N,)."""
        return self.offsets.diff() - 1

    @property
    def edep(self) -> Tensor:
        """The E_dep token's value of every event, (N,)."""
        return self.prefix[:, 1]

    def cond(self, name: str) -> Tensor:
        """Returns the conditioning scalar ``name`` of every event, (N,)."""
        return self.prefix[:, cond_slot(name)]

    @property
    def incoming(self) -> Tensor:
        """The incoming particle's token of every event, (N, 8)."""
        return self.tokens[self.offsets[:-1]]

    def species_counts(self, ptypes: Tensor) -> Tensor:
        """Counts the outgoing particles of each species per event, ``(N, len(ptypes))`` int16.

        ```
        counts[i, j] = the number of event i's outgoing tokens with pdgid ptypes[j]
        ```

        Raises:
            ValueError: ``ptypes`` holds pdgid 0, which is the padding's.
        """
        ptypes = torch.as_tensor(ptypes)
        if (ptypes == 0).any():
            raise ValueError("pdgid 0 is the padding's: a padded file would count its padded tokens")
        event, pid = self._outgoing()
        return torch.stack([torch.bincount(event[pid == p], minlength=len(self)).to(torch.int16) for p in ptypes], -1)

    def select(self, idx: Tensor) -> "PackedData":
        """Returns the events ``idx`` (a 1-D index, repeats allowed) as a packed dataset of their own.

        ```
        lens[k]                   = offsets[idx[k] + 1] - offsets[idx[k]]        new event k is old event idx[k]
        offsets'[k + 1]           = offsets'[k] + lens[k]
        tokens'[offsets'[k] + r]  = tokens[offsets[idx[k]] + r]                   r < lens[k]
        ```
        """
        idx = torch.as_tensor(idx, dtype=torch.long)
        lens = self.offsets[idx + 1] - self.offsets[idx]
        offsets = F.pad(lens.cumsum(0), (1, 0))
        src = (self.offsets[idx] - offsets[:-1]).repeat_interleave(lens) + torch.arange(int(offsets[-1]))
        return PackedData(prefix=self.prefix[idx], tokens=self.tokens[src], offsets=offsets, pad_e0=self.pad_e0[idx],
                          pad_token=self.pad_token, prefix_token=self.prefix_token, n_prefix=self.n_prefix, L=self.L,
                          edep_layers=None if self.edep_layers is None else self.edep_layers[idx])

    def unpack(self, idx=slice(None)) -> tuple:
        """Pads the events ``idx`` to ``L`` tokens.

        ```
        f[i, j]    = [prefix[i, j], *prefix_token[1:]]       j < n_prefix
                   = tokens[offsets[i] + j - n_prefix]       then the event's own lens[i] tokens
                   = [pad_e0[i], *pad_token[1:]]             then padding, up to L
        am[i, j]   = j < n_prefix + lens[i]                  lens[i] = offsets[i + 1] - offsets[i]
        m          = loss_mask(am, n_prefix)
        ```

        Args:
            idx: an index into the event axis (an int, a slice, an index tensor); what it selects is the batch shape.

        Returns:
            ``(f, m, am)``: (..., L, 8) float32, (..., L) int64 and (..., L) bool; the events' ``edep_layers`` as a
            4th tensor when the dataset carries it.
        """
        P, L = self.n_prefix, self.L
        start = self.offsets[:-1][idx]
        lens = self.offsets[1:][idx] - start
        f = torch.empty(*lens.shape, L, 8)
        f[..., :P, :] = self.prefix_token                  # the prefix
        f[..., :P, 0] = self.prefix[idx]
        f[..., P:, :] = self.pad_token                     # the rest as padding ...
        f[..., P:, 0] = self.pad_e0[idx].unsqueeze(-1)
        am = torch.arange(L) < (lens + P).unsqueeze(-1)
        valid = am[..., P:]                                # ... except the event's own tokens
        f[..., P:, :][valid] = self.tokens[(start.unsqueeze(-1) + torch.arange(L - P))[valid]]
        out = (f, loss_mask(am, P), am)
        return out if self.edep_layers is None else out + (self.edep_layers[idx],)

    def norm_e(self, prep) -> None:
        """Applies ``prep.norm_e`` to all events at once, in place: the result a padded file gets.

        ``norm_e`` changes the E_dep token (in the prefix), the incoming energy and ``edep_layers``, so only the prefix
        and incoming tokens are padded for it, and those three are written back.
        """
        P = self.n_prefix
        f, _, _, *edep_layers = prep.norm_e(self._prefix_and_incoming(), inplace=True)
        self.prefix = f[:, :P, 0].clone()
        self.tokens[self.offsets[:-1], 0] = f[:, P, 0]   # the incoming energy
        if edep_layers:
            self.edep_layers = edep_layers[0]

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
        d = token_names(d)   # legacy: a file that names the fields ``rows`` / ``pad_row`` / ``prefix_row``
        return cls(**{k: v for k, v in d.items() if k not in ("format", "version")})

    # ---- plumbing ------------------------------------------------------------------------------------------------

    def _outgoing(self) -> tuple[Tensor, Tensor]:
        """Returns each outgoing token's event index and pdgid, in token order."""
        ev = torch.arange(len(self)).repeat_interleave(self.offsets.diff())
        out = torch.ones(self.tokens.shape[0], dtype=torch.bool)
        out[self.offsets[:-1]] = False   # an event's first token is its incoming particle
        return ev[out], self.tokens[out, -1]

    def _prefix_and_incoming(self) -> tuple:
        """Returns every event's prefix and incoming tokens as a padded batch ``(f, None, None[, edep_layers])``.

        ``f`` is (N, n_prefix + 1, 8). The masks are None: ``norm_e`` does not read them, and for 20M events they
        take a GiB.
        """
        N, P = len(self), self.n_prefix
        v = torch.empty(N, P + 1, 8)
        v[:, :P] = self.prefix_token
        v[:, :P, 0] = self.prefix
        v[:, P] = self.incoming
        return (v, None, None) + (() if self.edep_layers is None else (self.edep_layers,))


class EventView:
    """The events ``idx`` of a dataset, without copying them: ``view[i]`` is ``ds[idx[i]]``."""

    def __init__(self, ds, idx: Tensor) -> None:
        self.ds, self.idx = ds, idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]


def interacting_only(ds) -> EventView:
    """Returns the view of ``ds`` without its pass-through events (no deposit and exactly one outgoing particle)."""
    return EventView(ds, ((ds.edep.flatten() > 0) | (ds.n_out != 1)).nonzero(as_tuple=True)[0])


def load_dataset(data: str | dict | tuple, *, prep=None, frac=None, device="cpu", **cut_args) -> PackedData:
    """Loads a dataset file, puts its MeV values on the model scale and keeps a fixed fraction of its events.

    Args:
        data:     a dataset directory (its packed file) or a ``.pt`` file, stored in MeV. Legacy: a padded file, the
            padded tensors in memory, or a raw dict (cut with ``cut_args``).
        prep:     the ``DataPrep`` whose ``norm_e`` puts the MeV values on the model scale; None: read as stored.
        frac:     the fraction of the events to keep, the same random subset every run; None: all of them.
        device:   of ``frac``'s generator.
        cut_args: ``GetLEGOData``'s arguments, for a legacy raw dict; a packed file ignores them.

    Returns:
        The events; a legacy input gives a ``PaddedData``, which answers the same calls.

    Raises:
        TypeError: a packed dataset not read from its file, or a ``prep`` without ``norm_e``.
    """
    from_file = isinstance(data, str)
    if from_file:
        data = torch.load(_dataset_file(data), map_location="cpu", weights_only=False)
    if not is_packed(data):   # legacy: padded tensors, raw particles
        return PaddedData.read(data, prep, frac, device, inplace=from_file, **cut_args)

    _check_packed_args(from_file, prep)
    events = PackedData.from_dict(data)
    if prep is not None:
        events.norm_e(prep)   # MeV -> the model scale
    if frac and frac < 1.0:
        events = events.select(_fixed_subset(len(events), frac, device))
    return events


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _fixed_subset(n: int, frac: float, device) -> Tensor:
    """Returns the indices of a random fraction ``frac`` of ``n`` events, the same every run."""
    idx = torch.randperm(n, device=device, generator=torch.Generator(device=device).manual_seed(0))[:int(n * frac)]
    return idx.cpu()


def _check_packed_args(from_file: bool, prep) -> None:
    """Raises if a packed dataset was not read from its file, or if ``prep`` has no ``norm_e``.

    Raises:
        TypeError: in either case.
    """
    if not from_file:
        raise TypeError("a packed dataset is read from its file: pass the path")
    if prep is not None and not hasattr(prep, "norm_e"):
        raise TypeError("a packed dataset takes a prep with norm_e (a DataPrep) or none")


def _dataset_file(directory: str) -> str:
    """Returns the dataset file a path names: a ``.pt`` file as it is, a directory's packed file, else its padded one.

    Raises:
        ValueError: the packed file is stale: it sits next to a padded one whose meta.json lacks ``"format": "packed"``
            (the padded file was written after it).
    """
    if directory.endswith(".pt"):
        return directory
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
