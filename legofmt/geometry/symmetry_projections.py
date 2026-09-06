"""``CubeSymmetry``: canonicalise directions onto a reference cube face.

Six rotations indexed by the face the position points at. Applied before the
flow and undone after it, and again in ``MultModel.proj_in``.
"""

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

    def __init__(self) -> None:
        self._cache: dict = {}

    def _rot(self, like: torch.Tensor) -> torch.Tensor:
        key = (like.device, like.dtype)
        r = self._cache.get(key)
        if r is None:
            r = self._cache[key] = self._R.to(like)
        return r

    def face_of(self, pos):
        axis = pos.abs().argmax(-1)
        neg = pos.gather(-1, axis.unsqueeze(-1)).squeeze(-1) < 0
        return 2 * axis + neg.long()

    def canonicalize(self, dirs, face):
        return torch.einsum("b...i,bij->b...j", dirs, self._rot(dirs)[face].mT)

    def uncanonicalize(self, dirs, face):
        return torch.einsum("b...i,bij->b...j", dirs, self._rot(dirs)[face])
