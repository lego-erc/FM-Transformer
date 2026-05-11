import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map


class _F:

    def __init__(self, f: Tensor) -> None:
        self.full = f

    @property
    def d(self) -> Tensor: return self.full[..., 0, 1]
    @property
    def edep(self) -> Tensor: return self.full[..., 1, 1]
    @property
    def pdgids(self) -> Tensor: return self.full[..., -1:]
    @property
    def non_p(self) -> Tensor: return self.full[..., :2, :]
    @property
    def in_p(self) -> Tensor: return self.full[..., 2:3, :]
    @property
    def out_p(self) -> Tensor: return self.full[..., 3:, :]
    @property
    def non_cc(self) -> Tensor: return self.full[..., :2, 1:7]
    @property
    def in_cc(self) -> Tensor: return self.full[..., 2:3, 1:7]
    @property
    def out_cc(self) -> Tensor: return self.full[..., 3:, 1:7]
    @property
    def model_in(self) -> Tensor: return self.full[..., 1:7]
    @property
    def mom(self) -> Tensor: return self.full[..., 1:4]


class _M:

    def __init__(self, m: Tensor) -> None:
        self.full = m.squeeze(-1) if m.ndim > 2 else m
    @property
    def out_p(self) -> Tensor: return self.full[..., 3:]


class DataStruct:

    def __init__(self, f: Tensor, m: Tensor, am: Tensor) -> None:
        self.f, self.m, self.am = _F(f), _M(m), _M(am)

    def __len__(self) -> int:
        return self.f.full.shape[0]

    def __getitem__(self, idx: int | Tensor) -> "DataStruct":
        return type(self)(self.f.full[idx], self.m.full[idx], self.am.full[idx])

    def to(self, *args, **kwargs) -> "DataStruct":
        return type(self)(
            self.f.full.to(*args, **kwargs),
            self.m.full.to(*args, **kwargs),
            self.am.full.to(*args, **kwargs),
        )

    @staticmethod
    def _collate(batch, *, collate_fn_map=None) -> "DataStruct":
        return DataStruct(
            torch.stack([s.f.full  for s in batch]),
            torch.stack([s.m.full  for s in batch]),
            torch.stack([s.am.full for s in batch]),
        )


default_collate_fn_map[DataStruct] = DataStruct._collate
