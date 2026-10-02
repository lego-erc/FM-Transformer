"""CubeSymmetry with the in-face stabiliser: the 8 elements are orthogonal, fix the canonical +x axis,
round-trip with the face rotation, keep the canonical face at +x, leave a normal-incidence gun at the
face centre invariant, and symmetrise a y-asymmetric cloud."""
import torch

from legofmt.geometry.symmetry_projections import CubeSymmetry


def test_stabiliser_matrices() -> None:
    S = CubeSymmetry._S
    assert torch.allclose(S @ S.mT, torch.eye(3).expand(8, 3, 3))
    assert torch.allclose(S @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]).expand(8, 3))
    assert len({tuple(m.flatten().tolist()) for m in S}) == 8


def test_round_trip_and_canonical_face() -> None:
    cs = CubeSymmetry()
    torch.manual_seed(0)
    dirs = torch.nn.functional.normalize(torch.randn(500, 5, 3), dim=-1)
    pos = torch.nn.functional.normalize(torch.randn(500, 3), dim=-1)
    face, g = cs.face_of(pos), torch.randint(8, (500,))
    assert torch.allclose(cs.uncanonicalize(cs.canonicalize(dirs, face, g), face, g), dirs, atol=1e-6)
    assert torch.allclose(cs.uncanonicalize(cs.canonicalize(dirs, face), face), dirs, atol=1e-6)
    pc = cs.canonicalize(pos.unsqueeze(1), face, g).squeeze(1)
    assert (pc.abs().argmax(-1) == 0).all() and (pc[:, 0] > 0).all()


def test_gun_is_a_fixed_point_and_cloud_is_symmetrised() -> None:
    cs = CubeSymmetry()
    gun = torch.tensor([[1.0, 0.0, 0.0]]).expand(8, 3)
    gc = cs.canonicalize(gun.unsqueeze(1), torch.zeros(8, dtype=torch.long), torch.arange(8)).squeeze(1)
    assert torch.allclose(gc, gun)
    torch.manual_seed(0)
    cloud = torch.randn(100000, 1, 3)
    cloud[..., 1] += 0.5
    face = torch.zeros(100000, dtype=torch.long)
    sym = cs.uncanonicalize(cs.canonicalize(cloud, face, torch.randint(8, (100000,))), face)
    assert abs(float(sym[..., 1].mean())) < 0.01 and abs(float(sym[..., 2].mean())) < 0.01
