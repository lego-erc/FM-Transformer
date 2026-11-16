"""Prepped datasets without the padding: each event's real rows plus where they start.

Everything but the particle rows of a padded ``(f, m, am[, edep_layers])`` follows from
a few per-event numbers: prefix rows are ``[value, *prefix_row[1:]]``, ``am`` is one
block from row 0, ``m`` is ``am`` with its first ``n_prefix + 1`` entries ``[0, 1, 0, ...]``,
and padded rows are ``[pad_e0, *pad_row[1:]]``. Those values are copied when packing, so
``unpack`` gives the padded tensors back bit for bit; ``from_padded`` refuses anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

PACKED_FORMAT = "lego-packed"
PACKED_VERSION = 1

_NAN_ROW = torch.full((8,), float("nan"))


def is_packed(obj) -> bool:
    return isinstance(obj, dict) and obj.get("format") == PACKED_FORMAT


def _mask(am: Tensor, n_prefix: int) -> Tensor:
    """The loss mask ``m``: ``am`` minus the prefix and incoming rows, plus the E_dep row (slot 1)."""
    m = am.long()
    m[..., :n_prefix + 1] = 0
    m[..., 1] = 1
    return m


def _merge_rows(rows: list[Tensor], name: str) -> Tensor:
    """The one known template row of several parts; all-NaN means unknown (a part with no such row)."""
    known = [r for r in rows if not torch.isnan(r).all()]
    for r in known[1:]:
        if not torch.equal(r[1:], known[0][1:]):
            raise ValueError(f"parts disagree on {name}")
    return known[0] if known else _NAN_ROW.clone()


@dataclass
class PackedData:
    prefix: Tensor       # (N, n_prefix) channel 0 of the prefix rows
    rows: Tensor         # (R, 8) every event's incoming and outgoing rows, back to back
    offsets: Tensor      # (N + 1,) event i's rows are rows[offsets[i]:offsets[i + 1]]
    pad_e0: Tensor       # (N,) channel 0 of an event's padded rows (NaN: none)
    pad_row: Tensor      # (8,) channels 1: of every padded row
    prefix_row: Tensor   # (8,) channels 1: of every prefix row
    n_prefix: int
    L: int
    edep_layers: Tensor | None = None

    @classmethod
    def from_padded(cls, f: Tensor, m: Tensor, am: Tensor, n_prefix: int,
                    edep_layers: Tensor | None = None) -> "PackedData":
        if f.dtype != torch.float32 or m.dtype != torch.int64 or am.dtype != torch.bool:
            raise TypeError(f"expected f float32, m int64, am bool; got {f.dtype}, {m.dtype}, {am.dtype}")
        N, L, _ = f.shape
        P = n_prefix
        tot = am.sum(1)
        if not torch.equal(am, torch.arange(L) < tot[:, None]) or (tot <= P).any():
            raise ValueError("am must be one block of at least n_prefix + 1 rows from row 0")
        if not torch.equal(m, _mask(am, P)):
            raise ValueError("m is not the pattern derived from am")
        prefix_row = f[0, 0].clone() if N else _NAN_ROW.clone()
        prefix_row[0] = float("nan")
        if not (f[:, :P, 1:] == prefix_row[1:]).all():
            raise ValueError("prefix rows differ outside channel 0")
        has_pad, pad = tot < L, ~am
        first = f[has_pad, tot[has_pad]]   # each padded event's first padded row
        pad_e0 = torch.full((N,), float("nan"))
        pad_e0[has_pad] = first[:, 0]
        pad_row = first[0].clone() if len(first) else _NAN_ROW.clone()
        pad_row[0] = float("nan")
        if not (f[pad][:, 1:] == pad_row[1:]).all():
            raise ValueError("padded rows differ outside channel 0")
        if not torch.equal(f[..., 0][pad], pad_e0.repeat_interleave(L - tot)):
            raise ValueError("an event's padded rows differ in channel 0")
        return cls(f[:, :P, 0].clone(), f[:, P:][am[:, P:]], F.pad((tot - P).cumsum(0), (1, 0)), pad_e0,
                   pad_row, prefix_row, P, L, None if edep_layers is None else edep_layers.clone())

    @staticmethod
    def concat(parts: list["PackedData"]) -> "PackedData":
        P, L = parts[0].n_prefix, parts[0].L
        if any(p.n_prefix != P or p.L != L for p in parts):
            raise ValueError("parts disagree on n_prefix or L")
        if len({p.edep_layers is None for p in parts}) > 1:
            raise ValueError("some parts carry edep_layers and some do not")
        cat = lambda name: torch.cat([getattr(p, name) for p in parts])
        offsets = F.pad(torch.cat([p.offsets.diff() for p in parts]).cumsum(0), (1, 0))
        return PackedData(cat("prefix"), cat("rows"), offsets, cat("pad_e0"),
                          _merge_rows([p.pad_row for p in parts], "pad_row"),
                          _merge_rows([p.prefix_row for p in parts], "prefix_row"), P, L,
                          None if parts[0].edep_layers is None else cat("edep_layers"))

    def __len__(self) -> int:
        return self.prefix.shape[0]

    @property
    def lengths(self) -> Tensor:
        """Rows per event in the padded layout (prefix + incoming + outgoing)."""
        return self.n_prefix + self.offsets.diff()

    @property
    def n_out(self) -> Tensor:
        return self.offsets.diff() - 1

    @property
    def incoming(self) -> Tensor:
        return self.rows[self.offsets[:-1]]

    def species_counts(self, ptypes: Tensor) -> Tensor:
        """Outgoing particles of each species per event, ``(N, len(ptypes))``."""
        ptypes = torch.as_tensor(ptypes)
        if (ptypes == 0).any():
            raise ValueError("pdgid 0 is the padding's: a padded file would count its padded rows")
        ev = torch.repeat_interleave(torch.arange(len(self)), self.offsets.diff())
        out = torch.ones(self.rows.shape[0], dtype=torch.bool)
        out[self.offsets[:-1]] = False
        pid, ev = self.rows[out, -1], ev[out]
        return torch.stack([torch.bincount(ev[pid == p], minlength=len(self)) for p in ptypes], -1)

    def select(self, idx: Tensor) -> "PackedData":
        idx = torch.as_tensor(idx, dtype=torch.long)
        lens = self.offsets[idx + 1] - self.offsets[idx]
        offsets = F.pad(lens.cumsum(0), (1, 0))
        src = torch.repeat_interleave(self.offsets[idx] - offsets[:-1], lens) + torch.arange(int(offsets[-1]))
        return PackedData(self.prefix[idx], self.rows[src], offsets, self.pad_e0[idx], self.pad_row,
                          self.prefix_row, self.n_prefix, self.L,
                          None if self.edep_layers is None else self.edep_layers[idx])

    def unpack(self, idx=slice(None)) -> tuple:
        """The padded ``(f, m, am[, edep_layers])`` of events ``idx`` (any index of the event axis), at full ``L``."""
        P, L = self.n_prefix, self.L
        start = self.offsets[:-1][idx]
        lens = self.offsets[1:][idx] - start
        f = torch.empty(*lens.shape, L, 8)
        f[..., :P, :] = self.prefix_row
        f[..., :P, 0] = self.prefix[idx]
        f[..., P:, :] = self.pad_row
        f[..., P:, 0] = self.pad_e0[idx][..., None]
        j = torch.arange(L - P)
        valid = j < lens[..., None]
        f[..., P:, :][valid] = self.rows[(start[..., None] + j)[valid]]
        am = torch.cat((torch.ones(*lens.shape, P, dtype=torch.bool), valid), -1)
        out = (f, _mask(am, P), am)
        return out if self.edep_layers is None else out + (self.edep_layers[idx],)

    def norm_e(self, prep) -> None:
        """``prep.norm_e`` in place, on the only rows it touches (prefix, incoming, edep_layers) of
        all events at once, as the padded file gets it."""
        N, P = len(self), self.n_prefix
        v = torch.empty(N, P + 1, 8)
        v[:, :P] = self.prefix_row
        v[:, :P, 0] = self.prefix
        v[:, P] = self.incoming
        batch = (v, torch.zeros(N, P + 1, dtype=torch.long), torch.ones(N, P + 1, dtype=torch.bool))
        out = prep.norm_e(batch + (() if self.edep_layers is None else (self.edep_layers,)), inplace=True)
        self.prefix = out[0][:, :P, 0].clone()
        self.rows[self.offsets[:-1], 0] = out[0][:, P, 0]
        if self.edep_layers is not None:
            self.edep_layers = out[3]

    def to_dict(self) -> dict:
        return {"format": PACKED_FORMAT, "version": PACKED_VERSION, **vars(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "PackedData":
        if not is_packed(d) or d.get("version") != PACKED_VERSION:
            raise ValueError(f"not a {PACKED_FORMAT} v{PACKED_VERSION} file: "
                             f"format={d.get('format')!r} version={d.get('version')!r}")
        return cls(**{k: v for k, v in d.items() if k not in ("format", "version")})
