"""Prepped datasets without the padding: each event's real tokens plus where they start.

Everything but the particle tokens of a padded ``(f, m, am[, edep_layers])`` follows from a few per-event numbers:

```
prefix tokens   [prefix[i, j], *prefix_token[1:]]
am              one block of n_prefix + 1 + n_out tokens from token 0
m               am, with its first n_prefix + 1 entries [0, 1, 0, ...]
padded tokens   [pad_e0[i], *pad_token[1:]]
```

Those values are copied when packing, so ``unpack`` gives the padded tensors back bit for bit; ``from_padded``
refuses anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

PACKED_FORMAT = "lego-packed"
PACKED_VERSION = 1

_NAN_TOKEN = torch.full((8,), float("nan"))


def is_packed(obj) -> bool:
    """Returns whether ``obj`` is a packed dataset as ``torch.load`` gives it (``PackedData.to_dict``)."""
    return isinstance(obj, dict) and obj.get("format") == PACKED_FORMAT


def _mask(am: Tensor, n_prefix: int) -> Tensor:
    """Builds the loss mask ``m``: ``am`` minus the prefix and incoming tokens, plus the E_dep token (slot 1)."""
    m = am.long()
    m[..., :n_prefix + 1] = 0
    m[..., 1] = 1
    return m


def _merge_tokens(tokens: list[Tensor], name: str) -> Tensor:
    """Returns the one known template token of several parts; all-NaN means unknown (a part with no such token)."""
    known = [r for r in tokens if not torch.isnan(r).all()]
    for r in known[1:]:
        if not torch.equal(r[1:], known[0][1:]):
            raise ValueError(f"parts disagree on {name}")
    return known[0] if known else _NAN_TOKEN.clone()


@dataclass
class PackedData:
    """A prepped dataset of N events without its padding; ``unpack`` pads events back to ``L`` tokens."""

    prefix: Tensor         # (N, n_prefix) channel 0 of the prefix tokens
    tokens: Tensor         # (R, 8) every event's incoming and outgoing tokens, back to back
    offsets: Tensor        # (N + 1,) event i's tokens are tokens[offsets[i]:offsets[i + 1]]
    pad_e0: Tensor         # (N,) channel 0 of an event's padded tokens (NaN: none)
    pad_token: Tensor      # (8,) channels 1: of every padded token
    prefix_token: Tensor   # (8,) channels 1: of every prefix token
    n_prefix: int
    L: int
    edep_layers: Tensor | None = None

    @classmethod
    def from_padded(cls, f: Tensor, m: Tensor, am: Tensor, n_prefix: int,
                    edep_layers: Tensor | None = None) -> "PackedData":
        """Packs padded tensors, refusing any that ``unpack`` would not give back bit for bit.

        Args:
            f:           (N, L, 8) float32, the features.
            m:           (N, L) int64, the loss mask.
            am:          (N, L) bool, the attention mask.
            n_prefix:    the number of prefix tokens.
            edep_layers: (N, n_layers), the per-layer E_dep, carried along.

        Raises:
            TypeError:  a tensor has another dtype.
            ValueError: ``am`` is not one block from token 0, ``m`` does not follow from ``am``, or the prefix /
                padded tokens are not one template outside channel 0.
        """
        if f.dtype != torch.float32 or m.dtype != torch.int64 or am.dtype != torch.bool:
            raise TypeError(f"expected f float32, m int64, am bool; got {f.dtype}, {m.dtype}, {am.dtype}")
        N, L, _ = f.shape
        P = n_prefix
        tot = am.sum(1)
        has_pad, pad = tot < L, ~am

        # the values everything but the particle tokens follows from
        prefix_token = f[0, 0].clone() if N else _NAN_TOKEN.clone()
        prefix_token[0] = float("nan")
        first = f[has_pad, tot[has_pad]]   # each padded event's first padded token
        pad_e0 = torch.full((N,), float("nan"))
        pad_e0[has_pad] = first[:, 0]
        pad_token = first[0].clone() if len(first) else _NAN_TOKEN.clone()
        pad_token[0] = float("nan")

        # ... which it does only if
        if not torch.equal(am, torch.arange(L) < tot[:, None]) or (tot <= P).any():
            raise ValueError("am must be one block of at least n_prefix + 1 tokens from token 0")
        if not torch.equal(m, _mask(am, P)):
            raise ValueError("m is not the pattern derived from am")
        if not (f[:, :P, 1:] == prefix_token[1:]).all():
            raise ValueError("prefix tokens differ outside channel 0")
        if not (f[pad][:, 1:] == pad_token[1:]).all():
            raise ValueError("padded tokens differ outside channel 0")
        if not torch.equal(f[..., 0][pad], pad_e0.repeat_interleave(L - tot)):
            raise ValueError("an event's padded tokens differ in channel 0")

        return cls(f[:, :P, 0].clone(), f[:, P:][am[:, P:]], F.pad((tot - P).cumsum(0), (1, 0)), pad_e0,
                   pad_token, prefix_token, P, L, None if edep_layers is None else edep_layers.clone())

    @staticmethod
    def concat(parts: list["PackedData"]) -> "PackedData":
        """Joins datasets event-wise, as packing their padded tensors in one go would.

        Raises:
            ValueError: the parts disagree on ``n_prefix``, ``L``, the template tokens, or on carrying
                ``edep_layers``.
        """
        P, L = parts[0].n_prefix, parts[0].L
        if any(p.n_prefix != P or p.L != L for p in parts):
            raise ValueError("parts disagree on n_prefix or L")
        if len({p.edep_layers is None for p in parts}) > 1:
            raise ValueError("some parts carry edep_layers and some do not")
        cat = lambda name: torch.cat([getattr(p, name) for p in parts])
        offsets = F.pad(torch.cat([p.offsets.diff() for p in parts]).cumsum(0), (1, 0))
        return PackedData(cat("prefix"), cat("tokens"), offsets, cat("pad_e0"),
                          _merge_tokens([p.pad_token for p in parts], "pad_token"),
                          _merge_tokens([p.prefix_token for p in parts], "prefix_token"), P, L,
                          None if parts[0].edep_layers is None else cat("edep_layers"))

    def __len__(self) -> int:
        return self.prefix.shape[0]

    @property
    def lengths(self) -> Tensor:
        """Tokens per event in the padded layout (prefix + incoming + outgoing)."""
        return self.n_prefix + self.offsets.diff()

    @property
    def n_out(self) -> Tensor:
        """Outgoing particles per event."""
        return self.offsets.diff() - 1

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
        out = (f, _mask(am, P), am)
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
        return {"format": PACKED_FORMAT, "version": PACKED_VERSION, **vars(self)}

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
