from torch import Tensor

class _T:

    def __init__(self, t: Tensor) -> None:
        self.full = t
    
    @property
    def dnst(self) -> Tensor:
        return self.full[:, 0, 1]
    
    @property
    def edep(self) -> Tensor:
        return self.full[:, 1, 1]
    
    @property
    def in_t(self) -> Tensor:
        return self.full[:, 2:3]
    
    @property
    def in_cc(self) -> Tensor:
        return self.full[:, 2:3, 1:7]
    
    @property
    def out_t(self) -> Tensor:
        return self.full[:, 3:]
    
    @property
    def out_cc(self) -> Tensor:
        return self.full[:, 3:, 1:7]


class DataStruct:

    def __init__(self, t: Tensor, m: Tensor, am: Tensor) -> None:
        self.t = _T(t)
        self.m = m
        self.am = am
