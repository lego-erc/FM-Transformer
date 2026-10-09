"""``CubeSymmetry``: canonicalise directions onto a reference cube face.

Six rotations indexed by the face the position points at. Applied before the
flow and undone after it. ``g`` (0..7)
additionally applies an element of the canonical face's stabiliser (the
dihedral group on the in-face axes y, z): drawn per event it is a training
augmentation, and at inference a random canonical frame, so the model's output
distribution carries the exact in-face symmetry of the data.
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

    # stabiliser of the canonical (+x) face: identity, 3 rotations about x, 4 reflections
    _S = torch.tensor([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
    ], dtype=torch.float32)

    def __init__(self) -> None:
        self._cache: dict = {}

    def _rot(self, like: torch.Tensor, name: str = "_R") -> torch.Tensor:
        key = (name, like.device, like.dtype)
        r = self._cache.get(key)
        if r is None:
            r = self._cache[key] = getattr(self, name).to(like)
        return r

    def _mat(self, like, face, g):
        r = self._rot(like)[face]
        return r if g is None else self._rot(like, "_S")[g] @ r

    def face_of(self, pos):
        axis = pos.abs().argmax(-1)
        neg = pos.gather(-1, axis.unsqueeze(-1)).squeeze(-1) < 0
        return 2 * axis + neg.long()

    def canonicalize(self, dirs, face, g=None):
        return torch.einsum("b...i,bij->b...j", dirs, self._mat(dirs, face, g).mT)

    def uncanonicalize(self, dirs, face, g=None):
        return torch.einsum("b...i,bij->b...j", dirs, self._mat(dirs, face, g))
