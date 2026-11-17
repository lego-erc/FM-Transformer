"""Packing prepped padded tensors into a ``PackedData`` (``legofmt.data.load.packed``): the step that writes a
``data_packed.pt``.

    pack     padded ``(f, m, am[, edep_layers])`` -> PackedData, refusing what ``unpack`` would not give back
    concat   PackedData parts -> one, as packing their padded tensors in one go would
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.data.load.packed import PackedData, loss_mask

_NAN_TOKEN = torch.full((8,), float("nan"))


def pack(f: Tensor, m: Tensor, am: Tensor, n_prefix: int, edep_layers: Tensor | None = None) -> PackedData:
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
    if not torch.equal(m, loss_mask(am, P)):
        raise ValueError("m is not the pattern derived from am")
    if not (f[:, :P, 1:] == prefix_token[1:]).all():
        raise ValueError("prefix tokens differ outside channel 0")
    if not (f[pad][:, 1:] == pad_token[1:]).all():
        raise ValueError("padded tokens differ outside channel 0")
    if not torch.equal(f[..., 0][pad], pad_e0.repeat_interleave(L - tot)):
        raise ValueError("an event's padded tokens differ in channel 0")

    return PackedData(f[:, :P, 0].clone(), f[:, P:][am[:, P:]], F.pad((tot - P).cumsum(0), (1, 0)), pad_e0,
                      pad_token, prefix_token, P, L, None if edep_layers is None else edep_layers.clone())


def concat(parts: list[PackedData]) -> PackedData:
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


def _merge_tokens(tokens: list[Tensor], name: str) -> Tensor:
    """Returns the one known template token of several parts; all-NaN means unknown (a part with no such token)."""
    known = [r for r in tokens if not torch.isnan(r).all()]
    for r in known[1:]:
        if not torch.equal(r[1:], known[0][1:]):
            raise ValueError(f"parts disagree on {name}")
    return known[0] if known else _NAN_TOKEN.clone()
