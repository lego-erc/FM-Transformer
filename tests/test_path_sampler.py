"""The closed-form geodesic path equals flow_matching's GeodesicProbPath, and the
sphere log map keeps its precision at small angles in fp32."""

import torch
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.cfm.path_sampler import ProductPathSampler
from legofmt.geometry.product_manifold import ProductManifold, sphere_logmap

MAN = ProductManifold([Euclidean(), Sphere(), Sphere()], (1, 3, 3))


def _unit(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape, 3, dtype=torch.float64), dim=-1)


def test_matches_geodesic_prob_path():
    torch.manual_seed(0)
    B, L = 64, 6
    x0 = torch.cat((torch.rand(B, L, 1, dtype=torch.float64), _unit(B, L), _unit(B, L)), -1)
    x1 = torch.cat((torch.rand(B, L, 1, dtype=torch.float64), _unit(B, L), _unit(B, L)), -1)
    x0[:, 0, 1:] = x1[:, 0, 1:] = 1.0  # the E_dep row's (1, 1, 1) direction filler
    t = torch.rand(B, dtype=torch.float64)
    new = ProductPathSampler(MAN).sample(x0, x1, t)
    tt = t.repeat_interleave(L)
    for sl, man in ((slice(0, 1), Euclidean()), (slice(1, 4), Sphere()), (slice(4, 7), Sphere())):
        ref = GeodesicProbPath(CondOTScheduler(), man).sample(
            x0[..., sl].flatten(0, 1), x1[..., sl].flatten(0, 1), tt)
        torch.testing.assert_close(new.x_t[..., sl].flatten(0, 1), ref.x_t, atol=1e-9, rtol=0)
        torch.testing.assert_close(new.dx_t[..., sl].flatten(0, 1), ref.dx_t, atol=1e-9, rtol=0)
    assert new.t.shape == (B, L) and (new.t == t[:, None]).all()


def test_sphere_logmap_small_angles_fp32():
    torch.manual_seed(0)
    x = _unit(4096)
    for theta in (3e-4, 1e-3, 1e-2, 1.0):
        v = torch.randn(4096, 3, dtype=torch.float64)
        v = theta * torch.nn.functional.normalize(v - (v * x).sum(-1, keepdim=True) * x, dim=-1)
        y32, x32 = Sphere().expmap(x, v).float(), x.float()
        ref = sphere_logmap(x32.double(), y32.double())
        err = ((sphere_logmap(x32, y32).double() - ref).norm(dim=-1) / theta).max()
        assert err < 1e-6, (theta, err)
        assert (MAN.logmap(torch.cat((x32[:, :1], x32, x32), -1), torch.cat((x32[:, :1], y32, y32), -1))[:, 1:4]
                == sphere_logmap(x32, y32)).all()
