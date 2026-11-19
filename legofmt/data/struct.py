"""The padded ``(B, L, 8)`` token layout, and the accessors that hide it.

```
tokens    [cond_scalars[0], E_dep, *cond_scalars[1:], incoming, outgoing_1, ..., outgoing_n, padding]
           `------------- prefix: n_prefix() -------'

columns   prefix token   [value, 1, 1, 1, 1, 1, 1, 0]
          particle       [energy, momentum direction (3), position direction (3), pdgid]

m         (B, L)   1 on the tokens the flow generates (E_dep, the outgoing particles), 0 on those it is given
am        (B, L)   True on a real token, False on padding
```

Where a token sits depends on ``cond_scalars``, so never index the sequence directly: use ``_F`` (the features),
``_M`` (a mask) and ``DataStruct`` (a batch). The layout is process-global module state set by ``set_layout``, so
constructing a model or a loader mutates it.
"""

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map


# The conditioning scalars, by their data_add keys; the default is the original (density, E_dep) prefix.
_COND_SCALARS: tuple[str, ...] = ("Density",)

# Where a particle token's channels start: energy | momentum direction (3) | position direction (3) | pdgid (absent
# from model_in); ``energy, mom, pos, pid = x.tensor_split(CHANNELS, dim=-1)``.
CHANNELS = (1, 4, 7)


def set_layout(cond_scalars) -> None:
    """Sets the process-global slot layout.

    Called by resolve_*_config and DataPrep, so constructing a model or a loader re-points every _F accessor.
    """
    global _COND_SCALARS
    _COND_SCALARS = tuple(cond_scalars)


def cond_scalars() -> tuple[str, ...]:
    """Returns the names of the conditioning scalars, in layout order."""
    return _COND_SCALARS


def n_prefix() -> int:
    """Returns the number of prefix tokens: the conditioning scalars and E_dep."""
    return len(_COND_SCALARS) + 1


def cond_slot(name: str) -> int:
    """Returns the token index of a conditioning scalar.

    cond_scalars[0] is token 0, the rest are shifted by one because token 1 is the generated E_dep.
    """
    if name not in _COND_SCALARS:
        raise ValueError(f"{name!r} not in cond_scalars {_COND_SCALARS}")
    i = _COND_SCALARS.index(name)
    return 0 if i == 0 else i + 1


class _F:
    """Named views into the features ``f`` (..., L, 8): writing through one writes ``f``."""

    def __init__(self, f: Tensor) -> None:
        self.full = f

    # The prefix scalars (...): the first conditioning scalar, E_dep, any conditioning scalar by name
    @property
    def d(self) -> Tensor: return self.full[..., 0, 0]
    @property
    def edep(self) -> Tensor: return self.full[..., 1, 0]
    def cond(self, name: str) -> Tensor: return self.full[..., cond_slot(name), 0]

    # Token groups: prefix (..., n_prefix, .), incoming (..., 1, .), outgoing (..., n_out, .); *_p with the pdgid
    # column, *_cc the 7 columns the manifold holds
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

    # Columns of every token (..., L, .)
    @property
    def model_in(self) -> Tensor: return self.full[..., 0:7]
    @property
    def energy(self) -> Tensor: return self.full[..., 0:1]
    @property
    def pdgids(self) -> Tensor: return self.full[..., -1:]

    # A particle's columns: the incoming particle (..., .) ...
    @property
    def in_energy(self) -> Tensor: return self.full[..., n_prefix(), 0:1]
    @property
    def in_mom(self) -> Tensor: return self.full[..., n_prefix(), 1:4]
    @property
    def in_pos(self) -> Tensor: return self.full[..., n_prefix(), 4:7]
    @property
    def in_pid(self) -> Tensor: return self.full[..., n_prefix(), -1]

    # ... and the outgoing particles (..., n_out, .)
    @property
    def out_energy(self) -> Tensor: return self.full[..., n_prefix() + 1:, 0]
    @property
    def out_mom(self) -> Tensor: return self.full[..., n_prefix() + 1:, 1:4]
    @property
    def out_pos(self) -> Tensor: return self.full[..., n_prefix() + 1:, 4:7]
    @property
    def out_pid(self) -> Tensor: return self.full[..., n_prefix() + 1:, -1]


class _M:
    """Named views into a mask (..., L): the loss mask ``m`` or the attention mask ``am``."""

    def __init__(self, m: Tensor) -> None:
        self.full = m.squeeze(-1) if m.ndim > 2 else m  # accepts (B, L) or (B, L, 1); callers mix the two

    @property
    def edep(self) -> Tensor: return self.full[..., 1]
    @property
    def out_p(self) -> Tensor: return self.full[..., n_prefix() + 1:]
    @property
    def fwd(self) -> Tensor: return self.full[..., n_prefix()] == 0   # of m: the incoming particle is given


class DataStruct:
    """A batch of events: the features ``f``, the loss mask ``m`` and the attention mask ``am``.

    Indexing, ``trim``, ``to`` and ``pin_memory`` apply to all of them and return a new ``DataStruct``.
    """

    def __init__(self, f: Tensor, m: Tensor, am: Tensor, edep_layers: Tensor | None = None) -> None:
        self.f, self.m, self.am = _F(f), _M(m), _M(am)
        self.edep_layers = edep_layers  # optional (B, n_layers) per-layer E_dep; no model reads it

    def __len__(self) -> int:
        return self.f.full.shape[0]

    def _map(self, fn) -> "DataStruct":
        """Applies ``fn`` to every tensor."""
        e = self.edep_layers
        return type(self)(fn(self.f.full), fn(self.m.full), fn(self.am.full), None if e is None else fn(e))

    def __getitem__(self, idx: int | Tensor) -> "DataStruct":
        return self._map(lambda t: t[idx])

    def trim(self, L: int) -> "DataStruct":
        """Cuts the token axis to its first ``L`` tokens."""
        return type(self)(self.f.full[:, :L], self.m.full[:, :L], self.am.full[:, :L], self.edep_layers)

    def pin_memory(self) -> "DataStruct":
        """Pins every tensor's memory: what a DataLoader with ``pin_memory`` calls on a batch."""
        return self._map(lambda t: t.contiguous().pin_memory())

    def to(self, *args, **kwargs) -> "DataStruct":
        """Moves every tensor as ``Tensor.to`` does, non-blocking unless told otherwise."""
        kwargs.setdefault("non_blocking", True)
        return self._map(lambda t: t.to(*args, **kwargs))

    @staticmethod
    def _collate(batch, *, collate_fn_map=None) -> "DataStruct":
        """Stacks single events to a batch: torch's default collate for a ``DataStruct``."""
        e = [s.edep_layers for s in batch]
        return DataStruct(
            torch.stack([s.f.full  for s in batch]),
            torch.stack([s.m.full  for s in batch]),
            torch.stack([s.am.full for s in batch]),
            None if e[0] is None else torch.stack(e),
        )


default_collate_fn_map[DataStruct] = DataStruct._collate
