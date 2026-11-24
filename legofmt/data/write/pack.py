"""Packing prepped padded tensors into a ``PackedData`` (``legofmt.data.load.packed``): the step before a
``data_packed.pt`` is written.

    pack     padded ``(f, m, am[, edep_layers])`` -> PackedData, refusing what ``unpack`` would not give back
    concat   PackedData parts -> one, as packing their padded tensors in one go would
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.data.load.packed import PackedData
from legofmt.data.struct import loss_mask

_NAN_TOKEN = torch.full((8,), float("nan"))


def pack(f: Tensor, m: Tensor, am: Tensor, n_prefix: int, edep_layers: Tensor | None = None,
         out_energy_mev: bool = True) -> PackedData:
    """Packs padded tensors, refusing any that ``unpack`` would not give back bit for bit.

    ```
    prefix[i]        = f[i, :n_prefix, 0]
    tokens           = f[i, n_prefix:tot[i]] of every event i, back to back         tot[i] = am[i].sum()
    offsets[i + 1]   = offsets[i] + tot[i] - n_prefix                                 offsets[0] = 0
    pad_e0[i]        = f[i, -1, 0] if event i has padding, else NaN
    prefix_token     = f[0, 0],                    channel 0 NaN
    pad_token        = the first padding token,    channel 0 NaN
    ```

    Args:
        f:           (N, L, 8) float32, the features; float64 also keeps the PDG ids exact (``pdgid``).
        m:           (N, L) int64, the loss mask.
        am:          (N, L) bool, the attention mask.
        n_prefix:    the number of prefix tokens.
        edep_layers: (N, n_layers), the per-layer E_dep, carried along; None: the dataset has none.
        out_energy_mev: the outgoing energy is in MeV, as ``DataPrep.prep`` writes it; false: the tensors of a file
            from before, on the legacy log scale.

    Raises:
        TypeError:  a tensor has another dtype.
        ValueError: ``am`` is not one block of at least ``n_prefix + 1`` tokens from token 0, ``m`` does not follow
            from ``am``, the prefix or the padding tokens are not one template outside channel 0, or an event's
            padding tokens differ in channel 0.
    """
    pdgid = f[:, n_prefix:][am[:, n_prefix:]][:, 7].round().int() if f.dtype == torch.float64 else None
    f = f.float()   # the tokens are float32 either way
    _check_packable(f, m, am, n_prefix)
    P, tot = n_prefix, am.sum(1)
    prefix_token, pad_token, pad_e0 = _templates(f, tot)
    return PackedData(
        prefix=f[:, :P, 0].clone(),
        tokens=f[:, P:][am[:, P:]],
        offsets=F.pad((tot - P).cumsum(0), (1, 0)),
        pad_e0=pad_e0,
        pad_token=pad_token,
        prefix_token=prefix_token,
        n_prefix=P,
        L=f.shape[1],
        edep_layers=None if edep_layers is None else edep_layers.clone(),
        out_energy_mev=out_energy_mev,
        pdgid=pdgid,
    )


def concat(parts: list[PackedData]) -> PackedData:
    """Joins datasets event-wise, as packing their padded tensors in one go would.

    Raises:
        ValueError: the parts disagree on ``n_prefix``, ``L``, the template tokens, on carrying ``edep_layers`` or
            exact PDG ids, or on the unit of the outgoing energy.
    """
    P, L = parts[0].n_prefix, parts[0].L
    if any(p.n_prefix != P or p.L != L for p in parts):
        raise ValueError("parts disagree on n_prefix or L")
    if len({p.edep_layers is None for p in parts}) > 1:
        raise ValueError("some parts carry edep_layers and some do not")
    if len({p.out_energy_mev for p in parts}) > 1:
        raise ValueError("some parts store the outgoing energy in MeV and some on the legacy log scale")
    if len({p.pdgid is None for p in parts}) > 1:
        raise ValueError("some parts carry exact PDG ids and some do not")
    cat = lambda name: torch.cat([getattr(p, name) for p in parts])
    offsets = F.pad(torch.cat([p.offsets.diff() for p in parts]).cumsum(0), (1, 0))
    return PackedData(prefix=cat("prefix"), tokens=cat("tokens"), offsets=offsets, pad_e0=cat("pad_e0"),
                      pad_token=_merge_tokens([p.pad_token for p in parts], "pad_token"),
                      prefix_token=_merge_tokens([p.prefix_token for p in parts], "prefix_token"), n_prefix=P, L=L,
                      edep_layers=None if parts[0].edep_layers is None else cat("edep_layers"),
                      out_energy_mev=parts[0].out_energy_mev, pdgid=None if parts[0].pdgid is None else cat("pdgid"))


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _templates(f: Tensor, tot: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Returns ``(prefix_token, pad_token, pad_e0)``, the values the prefix and padding tokens follow from.

    A template token has channel 0 NaN, and is all NaN where there is no such token (no event, no event with padding).
    """
    N, L, _ = f.shape
    has_pad = tot < L
    first_pad = f[has_pad, tot[has_pad]]   # the first padding token of each event that has padding
    prefix_token = f[0, 0].clone() if N else _NAN_TOKEN.clone()
    prefix_token[0] = float("nan")
    pad_token = first_pad[0].clone() if len(first_pad) else _NAN_TOKEN.clone()
    pad_token[0] = float("nan")
    pad_e0 = torch.where(has_pad, f[:, -1, 0], float("nan"))   # the last token is padding if any is
    return prefix_token, pad_token, pad_e0


def _check_packable(f: Tensor, m: Tensor, am: Tensor, n_prefix: int) -> None:
    """Raises unless ``unpack`` gives the padded tensors back bit for bit: the errors that ``pack`` lists."""
    if f.dtype != torch.float32 or m.dtype != torch.int64 or am.dtype != torch.bool:
        raise TypeError(f"expected f float32, m int64, am bool; got {f.dtype}, {m.dtype}, {am.dtype}")
    L, P = f.shape[1], n_prefix
    tot, pad = am.sum(1), ~am
    prefix_token, pad_token, pad_e0 = _templates(f, tot)
    if not torch.equal(am, torch.arange(L) < tot.unsqueeze(1)) or (tot <= P).any():
        raise ValueError("am must be one block of at least n_prefix + 1 tokens from token 0")
    if not torch.equal(m, loss_mask(am, P)):
        raise ValueError("m is not the pattern derived from am")
    if not (f[:, :P, 1:] == prefix_token[1:]).all():
        raise ValueError("prefix tokens differ outside channel 0")
    if not (f[pad][:, 1:] == pad_token[1:]).all():
        raise ValueError("padded tokens differ outside channel 0")
    if not torch.equal(f[..., 0][pad], pad_e0.repeat_interleave(L - tot)):
        raise ValueError("an event's padded tokens differ in channel 0")


def _merge_tokens(tokens: list[Tensor], name: str) -> Tensor:
    """Returns the template token the parts share; all NaN if no part has such a token.

    A part without such a token holds it all NaN and does not count.

    Raises:
        ValueError: two parts know different templates.
    """
    known = [r for r in tokens if not torch.isnan(r).all()]
    for r in known[1:]:
        if not torch.equal(r[1:], known[0][1:]):
            raise ValueError(f"parts disagree on {name}")
    return known[0] if known else _NAN_TOKEN.clone()
