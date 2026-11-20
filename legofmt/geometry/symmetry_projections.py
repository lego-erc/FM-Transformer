"""``CubeSymmetry``: canonicalise directions onto a reference cube face.

Six rotations indexed by the face the position points at. Applied before the
flow and undone after it, and again in ``MultModel.proj_in``. ``g`` (0..7)
additionally applies an element of the canonical face's stabiliser (the
dihedral group on the in-face axes y, z): drawn per event it is a training
augmentation, and at inference a random canonical frame, so the model's output
distribution carries the exact in-face symmetry of the data.
"""

import torch


class CubeSymmetry:
    """The cube rotations that bring a face onto the canonical (+x) one, and that face's stabiliser."""

    # one rotation per face, indexed 2 * axis + (the face is on the negative side)
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

    def _mat(self, like, face, g):
        """Returns each event's matrix: its face's rotation, then the stabiliser element ``g`` if given."""
        r = self._rot(like)[face]
        return r if g is None else self._rot(like, "_S")[g] @ r

    def face_of(self, pos):
        """Returns the face a position direction points at: 2 * (its dominant axis) + (it is negative there)."""
        axis = torch.abs(pos).argmax(-1)
        neg  = _component(pos, axis) < 0
        return 2 * axis + neg.long()

    def canonicalize(self, dirs, face, g=None):
        """Rotates ``dirs`` (B, ..., 3) into the canonical frame of each event's ``face`` (and by ``g``)."""
        return torch.einsum("b...i,bij->b...j", dirs, self._mat(dirs, face, g).mT)

    def uncanonicalize(self, dirs, face, g=None):
        """Inverts ``canonicalize``."""
        return torch.einsum("b...i,bij->b...j", dirs, self._mat(dirs, face, g))

    def _rot(self, like: torch.Tensor, name: str = "_R") -> torch.Tensor:
        """Returns the matrix table ``name`` on the device and dtype of ``like`` (cached)."""
        key = (name, like.device, like.dtype)
        r = self._cache.get(key)
        if r is None:
            r = self._cache[key] = getattr(self, name).to(like)
        return r


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _component(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Returns each vector's component ``idx``: ``v[..., idx]`` per row."""
    return v.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
