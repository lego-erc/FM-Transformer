"""The padded ``(B, L, 8)`` sequence layout, and the accessors that hide it.

Slot roles depend on ``cond_scalars``, so never index the sequence directly --
use ``_F`` (``d``, ``edep``, ``cond(name)``, ``in_p``, ``out_p``, ``model_in``,
``energy``, ``pdgids``) and ``DataStruct``. The layout is process-global module
state set by ``set_layout``, so constructing a model mutates it.
"""

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map


# Conditioning-scalar slot layout: [cond[0]=density, edep, *cond[1:], incoming,
# *outgoing]. Names are data_add keys; default = original (density, edep) prefix.
_COND_SCALARS: tuple[str, ...] = ("Density",)


def set_layout(cond_scalars) -> None:
    """Set the process-global slot layout. Called by resolve_*_config and
    DataPrep, so constructing a model or a loader re-points every _F accessor."""
    global _COND_SCALARS
    _COND_SCALARS = tuple(cond_scalars)


def cond_scalars() -> tuple[str, ...]:
    return _COND_SCALARS


def n_prefix() -> int:
    return len(_COND_SCALARS) + 1


def cond_slot(name: str) -> int:
    """Row index of a conditioning scalar: cond_scalars[0] is row 0, the rest
    are shifted by one because row 1 is the generated E_dep."""
    if name not in _COND_SCALARS:
        raise ValueError(f"{name!r} not in cond_scalars {_COND_SCALARS}")
    i = _COND_SCALARS.index(name)
    return 0 if i == 0 else i + 1


class _F:

    def __init__(self, f: Tensor) -> None:
        self.full = f

    @property
    def d(self) -> Tensor: return self.full[..., 0, 0]
    @property
    def edep(self) -> Tensor: return self.full[..., 1, 0]
    def cond(self, name: str) -> Tensor: return self.full[..., cond_slot(name), 0]
    @property
    def pdgids(self) -> Tensor: return self.full[..., -1:]
    @property
    def non_p(self) -> Tensor: return self.full[..., :n_prefix(), :]
    @property
    def in_p(self) -> Tensor: return self.full[..., n_prefix():n_prefix() + 1, :]
    @property
    def out_p(self) -> Tensor: return self.full[..., n_prefix() + 1:, :]
    @property
    def non_cc(self) -> Tensor: return self.full[..., :n_prefix(), 0:7]
    @property
    def in_cc(self) -> Tensor: return self.full[..., n_prefix():n_prefix() + 1, 0:7]
    @property
    def out_cc(self) -> Tensor: return self.full[..., n_prefix() + 1:, 0:7]
    @property
    def model_in(self) -> Tensor: return self.full[..., 0:7]
    @property
    def energy(self) -> Tensor: return self.full[..., 0:1]


class _M:

    def __init__(self, m: Tensor) -> None:
        self.full = m.squeeze(-1) if m.ndim > 2 else m  # accepts (B, L) or (B, L, 1); callers mix the two
    @property
    def out_p(self) -> Tensor: return self.full[..., n_prefix() + 1:]


class DataStruct:

    def __init__(self, f: Tensor, m: Tensor, am: Tensor, edep_layers: Tensor | None = None) -> None:
        self.f, self.m, self.am = _F(f), _M(m), _M(am)
        self.edep_layers = edep_layers  # optional (B, n_layers) per-layer E_dep; no model reads it

    def __len__(self) -> int:
        return self.f.full.shape[0]

    def _map(self, fn) -> "DataStruct":
        e = self.edep_layers
        return type(self)(fn(self.f.full), fn(self.m.full), fn(self.am.full), None if e is None else fn(e))

    def __getitem__(self, idx: int | Tensor) -> "DataStruct":
        return self._map(lambda t: t[idx])

    def trim(self, L: int) -> "DataStruct":
        return type(self)(self.f.full[:, :L], self.m.full[:, :L], self.am.full[:, :L], self.edep_layers)

    def pin_memory(self) -> "DataStruct":
        return self._map(lambda t: t.contiguous().pin_memory())

    def to(self, *args, **kwargs) -> "DataStruct":
        kwargs.setdefault("non_blocking", True)
        return self._map(lambda t: t.to(*args, **kwargs))

    @staticmethod
    def _collate(batch, *, collate_fn_map=None) -> "DataStruct":
        e = [s.edep_layers for s in batch]
        return DataStruct(
            torch.stack([s.f.full  for s in batch]),
            torch.stack([s.m.full  for s in batch]),
            torch.stack([s.am.full for s in batch]),
            None if e[0] is None else torch.stack(e),
        )


default_collate_fn_map[DataStruct] = DataStruct._collate
