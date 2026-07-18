import torch


class CubeSymmetry:
    _R = torch.tensor([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
    ], dtype=torch.float32)

    def face_of(self, pos):
        axis = pos.abs().argmax(-1)
        neg = pos.gather(-1, axis.unsqueeze(-1)).squeeze(-1) < 0
        return 2 * axis + neg.long()

    def canonicalize(self, dirs, face):
        return torch.einsum("b...i,bij->b...j", dirs, self._R.to(dirs)[face].mT)

    def uncanonicalize(self, dirs, face):
        return torch.einsum("b...i,bij->b...j", dirs, self._R.to(dirs)[face])
